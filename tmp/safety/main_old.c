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
static K_THREAD_STACK_DEFINE(tx_stack, 2048);
static struct k_thread tx_thread_data;
static void tx_thread(void);
static void init_tx_thrad();
static int sent_packets = 0;
static int received_packets = 0;
// TODO: Use mac?
uint8_t *mac_addr; // 8 bytes MAC
//static struct net_pkt *pkt_curr;
#define TX_THREAD_PRIORITY K_HIGHEST_THREAD_PRIO

// measurement
// 40 bit measurements
// we save each measurement

// someone sent a message or someone received a message
struct __attribute__((__packed__)) msg_ts {
    uint8_t addr[2];
    uint8_t sn;
    uint8_t ts[5];
};

// TODO: This is not standard compliant
static uint8_t msg_header[] = {0xDE, 0xCA};

// we keep things simple and define a buffer for the timestamps
// at position 0 is our tx timestamp, others are rx timestamps (or just zero if not used)
#define NUM_MSG_TS ((127-sizeof(msg_header))/sizeof(struct msg_ts))

static struct msg_ts msg_ts_buf[NUM_MSG_TS];

K_SEM_DEFINE(msg_ts_buf_sem, 1, 1);

static uint16_t next_rx_write_pos = 0;

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

    // reset msg buffer
    (void)memset(msg_ts_buf, 0, sizeof(msg_ts_buf));

    msg_ts_buf[0].addr[0] = 0x00;
    msg_ts_buf[0].addr[1] = 0x01;

    mac_addr = dwt_get_mac(ieee802154_dev);
    // we disable the frame filter, otherwise the packets are not received!
    dwt_set_frame_filter(ieee802154_dev, 0, 0);

    radio_api = (struct ieee802154_radio_api *)ieee802154_dev->api;

    LOG_INF("Start IEEE 802.15.4 device");
    ret = radio_api->start(ieee802154_dev);
    if(ret) {
        LOG_ERR("Could not start ieee 802.15.4 device");
        return false;
    }


    // Create TX thread and queue
    init_tx_thrad();

    while (1) {
        if (sent_packets > 0 && sent_packets % 100 == 0) {
            LOG_DBG("Sent %d packets", sent_packets);
        }

        if (received_packets > 0 && received_packets % 100 == 0) {
            LOG_DBG("Received %d packets", sent_packets);
        }

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
    struct net_buf *buf = pkt->buffer;
    int ret;

    received_packets++;

    //if (len > sizeof(msg_header) && !memcmp(msg_header, net_buf_pull_mem(buf, sizeof(msg_header)), sizeof(msg_header))) {
    //LOG_WRN("Got data, pkt %p, len %d", pkt, len);
    //} else {
    //    LOG_DBG("Got WRONG data, pkt %p, len %d", pkt, len);
    //}

    net_pkt_hexdump(pkt, "<");

    out:
    net_pkt_unref(pkt);

    return ret;
}


static void init_tx_thrad(void)
{
    /* Transmit queue init */
    k_thread_create(&tx_thread_data, tx_stack,
                    K_THREAD_STACK_SIZEOF(tx_stack),
                    (k_thread_entry_t)tx_thread,
                    NULL, NULL, NULL, TX_THREAD_PRIORITY, 0, K_NO_WAIT);
}

static void tx_thread(void)
{
    LOG_DBG("TX thread started");

    uint8_t own_addr[2];
    uint8_t sn = 0;

    while (true) {

        struct net_pkt *pkt = NULL;
        struct net_buf *buf = NULL;
        size_t len = 4;//sizeof (msg_header)+sizeof(msg_ts_buf);

        while(pkt == NULL) {
            pkt = net_pkt_alloc_with_buffer(NULL, len, AF_UNSPEC, 0,K_MSEC(100));//K_NO_WAIT);
            if (!pkt) {
                LOG_WRN("COULD NOT ALLOCATE MEMORY FOR PACKET!");
            }
        }

        net_pkt_write_u8(pkt, 1);

        // prepare and transmit the packet
        {

            //LOG_DBG("Send pkt %p buf %p len %d", pkt, buf, buf->len);
            //LOG_HEXDUMP_DBG(buf->data, buf->len, "TX Data");

            int retries = 1; //TODO: the tx_ts needs to be updated for the retries!
            int ret;

            uint64_t uus_delay = debug_transmission ? 1000000 : 50000; // what value to choose here? Depends on the processor etc! (and also if we debug stuff)
            uint64_t estimated_ts = 0;

            struct net_ptp_time ts;
            ts.second = 0;
            ts.nanosecond = 0;
            net_pkt_set_timestamp(pkt, &ts);

            net_pkt_write(pkt, msg_header, sizeof(msg_header));
            k_sem_take(&msg_ts_buf_sem, K_FOREVER);
            do {

                //LOG_DBG("TX start us: %llu", dwt_ts_to_fs(dwt_system_ts(ieee802154_dev)) / 1000000000U);
                //uint64_t start = debug_transmission ? dwt_system_ts(ieee802154_dev) : 0;
                uint8_t dst[8];
                estimated_ts = dwt_plan_delayed_tx(ieee802154_dev, uus_delay);
                sys_put_le64(estimated_ts, dst);

                for(int i = 0; i < sizeof(msg_ts_buf[0].ts); i++) {
                    msg_ts_buf[0].ts[i] = dst[i];
                }

                // all other entries are updated in the rx event!
                net_pkt_write(pkt, msg_ts_buf, sizeof(msg_ts_buf));

                //uint64_t end = debug_transmission ? dwt_system_ts(ieee802154_dev) : 0;
                ret = radio_api->tx(ieee802154_dev, IEEE802154_TX_MODE_TXTIME, pkt, buf);

                //LOG_DBG("us duration: %llu", dwt_ts_to_fs(end-start) / 1000000000U);
                //LOG_DBG("us time: %llu", dwt_ts_to_fs(dwt_system_ts(ieee802154_dev)) / 1000000000U);
                if(!ret) {
                    msg_ts_buf[0].sn++;
                }

            } while (ret && retries--);
            k_sem_give(&msg_ts_buf_sem);

            if (ret) {
                LOG_ERR("TX: Error transmitting data!");
            } else {

                //LOG_DBG("Sent pkt %p len %d", pkt, len);
                sent_packets++;

                uint64_t estimated_us = dwt_ts_to_fs(estimated_ts) / 1000000000U;
                uint64_t estimated_ns = dwt_ts_to_fs(estimated_ts) / 1000000U;

                struct net_ptp_time *actual_ts = net_pkt_timestamp(pkt);
                uint64_t actual_ns = actual_ts->second * 1000000000U + actual_ts->nanosecond;
                //LOG_DBG("TX: Estimated us %llu", estimated_us);
                //LOG_DBG("TX: Estimated us %llu Actual ns %llu", estimated_ns, actual_ns);
            }
        }

        net_pkt_unref(pkt);
    }
}