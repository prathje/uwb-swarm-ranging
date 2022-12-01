#include <logging/log.h>
#include <zephyr.h>

#include <net/net_core.h>
#include <net/ieee802154_radio.h>
#include <drivers/ieee802154/dw1000.h>

LOG_MODULE_REGISTER(main);


/* ieee802.15.4 device */
static struct ieee802154_radio_api *radio_api;
static const struct device *ieee802154_dev;

/**
 * Stack for the tx thread.
 */
static K_THREAD_STACK_DEFINE(tx_stack, 1024);
static struct k_thread tx_thread_data;
static void tx_thread(void);
static void init_tx_queue();
static struct k_fifo tx_queue;


static int sent_packets = 0;
// TODO: Use mac?
//uint8_t mac_addr[8];
//static struct net_pkt *pkt_curr;

#define TX_THREAD_PRIORITY K_PRIO_COOP(CONFIG_NUM_COOP_PRIORITIES - 1)



enum net_verdict ieee802154_radio_handle_ack(struct net_if *iface, struct net_pkt *pkt)
{
    return NET_CONTINUE;
}

int main(void) {
    int ret = 0;
    LOG_INF("Starting ...");

    LOG_INF("Initialize ieee802.15.4");
    ieee802154_dev = device_get_binding(CONFIG_NET_CONFIG_IEEE802154_DEV_NAME);
    if (!ieee802154_dev) {
        LOG_ERR("Cannot get ieee 802.15.4 device");
        return false;
    }

    radio_api = (struct ieee802154_radio_api *)ieee802154_dev->api;

    LOG_INF("Start IEEE 802.15.4 device");
    ret = radio_api->start(ieee802154_dev);
    if(ret) {
        LOG_ERR("Could not start ieee 802.15.4 device");
        return false;
    }


    // Create TX thread and queue
    init_tx_queue();

    while (1) {
        // send a dummy packet for now!
        struct net_pkt *pkt;

        int buf_len = 4;
        /* Maximum 2 bytes are added to the len */
        pkt = net_pkt_alloc_with_buffer(NULL, buf_len, AF_UNSPEC, 0,
                                        K_MSEC(10));//K_NO_WAIT);
        if (pkt) {
            net_pkt_write_u8(pkt, 1);
            k_fifo_put(&tx_queue, pkt);
        } else {
            //TODO: Enable again LOG_ERR("COULD NOT ALLOCATE MEMORY FOR PACKET!");
        }

        if (sent_packets > 0 && sent_packets % 1000 == 0) {
            LOG_DBG("Sent %d packets", sent_packets);
        }

        k_msleep(2);

        k_yield();
    }
    return 0;
}


static void net_pkt_hexdump(struct net_pkt *pkt, const char *str)
{
    struct net_buf *buf = pkt->buffer;

    while (buf) {
        LOG_HEXDUMP_DBG(buf->data, buf->len, str);
        buf = buf->frags;
    }
}

/**
 * Interface to the network stack, will be called when the packet is
 * received
 */
int net_recv_data(struct net_if *iface, struct net_pkt *pkt)
{
    size_t len = net_pkt_get_len(pkt);
    int ret;

    LOG_DBG("Got data, pkt %p, len %d", pkt, len);

    net_pkt_hexdump(pkt, "<");

    out:
    net_pkt_unref(pkt);

    return ret;
}


static void init_tx_queue(void)
{
    /* Transmit queue init */
    k_fifo_init(&tx_queue);

    k_thread_create(&tx_thread_data, tx_stack,
                    K_THREAD_STACK_SIZEOF(tx_stack),
                    (k_thread_entry_t)tx_thread,
                    NULL, NULL, NULL, TX_THREAD_PRIORITY, 0, K_NO_WAIT);
}


static void tx_thread(void)
{
    LOG_DBG("TX thread started");

    while (true) {
        struct net_pkt *pkt;
        struct net_buf *buf;
        size_t len;

        pkt = k_fifo_get(&tx_queue, K_FOREVER);
        buf = net_buf_frag_last(pkt->buffer);
        len = net_pkt_get_len(pkt);

        //LOG_DBG("Send pkt %p buf %p len %d", pkt, buf, len);

        //LOG_HEXDUMP_DBG(buf->data, buf->len, "TX Data");

        // transmit the packet
        {
            int retries = 3;
            int ret;

            uint64_t uus_delay = 80000000; // what value to choose here? Depends on the processor etc!
            uint64_t estimated_ts = 0;

            struct net_ptp_time ts;
            ts.second = 0;
            ts.nanosecond = 0;
            net_pkt_set_timestamp(pkt, &ts);

            do {
                estimated_ts = dwt_plan_delayed_tx(ieee802154_dev, uus_delay);
                ret = radio_api->tx(ieee802154_dev, IEEE802154_TX_MODE_TXTIME, pkt, buf);
            } while (ret && retries--);

            if (ret) {
                LOG_ERR("TX: Error transmitting data!");
            } else {
                sent_packets++;

                uint64_t estimated_ns = dwt_ts_to_fs(estimated_ts) / 1000000U;

                struct net_ptp_time *actual_ts = net_pkt_timestamp(pkt);
                uint64_t actual_ns = actual_ts->second * 1000000000U + actual_ts->nanosecond;

                //LOG_DBG("TX: Estimated %llu Actual %llu", estimated_ns, actual_ns);
            }
        }

        net_pkt_unref(pkt);
    }
}