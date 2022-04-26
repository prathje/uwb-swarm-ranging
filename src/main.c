#include <logging/log.h>
#include <zephyr.h>

#include <net/net_core.h>
#include <net/ieee802154_radio.h>
#include <drivers/ieee802154/dw1000.h>
#include <stdio.h>
#include <drivers/uart.h>

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
//static struct net_pkt *pkt_curr;

#define TX_THREAD_PRIORITY K_PRIO_COOP(CONFIG_NUM_COOP_PRIORITIES - 1)

// measurement
// 40 bit measurements
// we save each measurement

// someone sent a message or someone received a message
struct __attribute__((__packed__)) msg_ts {
    uint8_t addr[2];
    uint8_t sn;
    uint8_t ts[5];
};

// TODO: Use mac?
uint8_t *mac_addr; // 8 bytes MAC

// TODO: This is not standard compliant
static uint8_t msg_header[] = {0xDE, 0xCA};

// we keep things simple and define a buffer for the timestamps
// at position 0 is our tx timestamp, others are rx timestamps (or just zero if not used)
#define NUM_MSG_TS ((127-sizeof(msg_header))/sizeof(struct msg_ts))

//used to build the tx msgs
static struct msg_ts msg_tx_buf[NUM_MSG_TS];
static uint64_t num_receptions = 0; // used to determine the amount of rx timestamps and also the next write pos
K_SEM_DEFINE(msg_tx_buf_sem, 0, 1);

static uint16_t next_rx_write_pos = 0;

static const struct device* uart_device = DEVICE_DT_GET(DT_CHOSEN(zephyr_console));
static void uart_out(char* msg) {
    while (*msg != '\0') {
        uart_poll_out(uart_device, *msg);
        msg++;
    }
}

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

    // prepare msg buffer
    mac_addr = dwt_get_mac(ieee802154_dev);
    {
        (void)memset(msg_tx_buf, 0, sizeof(msg_tx_buf));
        // TODO: Add own addr!
        msg_tx_buf[0].addr[0] = mac_addr[1];
        msg_tx_buf[0].addr[1] = mac_addr[2];

        k_sem_give(&msg_tx_buf_sem);
    }

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
    init_tx_queue();

    while (1) {
        // send a dummy packet for now!

        if (sent_packets > 0 && sent_packets % 1000 == 0) {
            LOG_DBG("Sent %d packets", sent_packets);
        }

        k_msleep(1);

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

static int format_msg_ts_to_json(char *buf, size_t buf_len, struct msg_ts *msg_ts) {
    uint16_t addr = sys_get_le16(&msg_ts->addr[0]);
    uint64_t ts = ((uint64_t)sys_get_le32(&msg_ts->ts[1]) << 8) | msg_ts->ts[0];
    return snprintf(buf, buf_len, "{\"addr\": \"0x%04X\", \"sn\": %d, \"ts\": %llu}", addr, msg_ts->sn, ts);
}

static void output_msg_to_uart(char* type, struct msg_ts msg_ts_arr[], size_t num_ts, float *clock_ratio_offset) {
    if (num_ts == 0) {
        uart_out("{}");
        return;
    }

    // write open parentheses
    uart_out("{ \"type\": \"");
    uart_out(type);
    uart_out("\", ");

    char buf[256];

    if (clock_ratio_offset != NULL) {
        snprintf(buf, sizeof(buf), "\"clock_ratio_offset\": %f", *clock_ratio_offset);
        uart_out(buf);
    }


    uart_out(", \"tx\": ");


    int ret = 0;
    ret = format_msg_ts_to_json(buf, sizeof(buf), &msg_ts_arr[0]);

    if (ret < 0) {
        uart_out("\n");
        return;
    }

    uart_out(buf);
    uart_out(", \"rx\": [");

    // write message content
    for(int i = 1; i < num_ts; i++) {
        // add separators in between
        if (i > 1)
        {
            uart_out(", ");
        }

        // write ts content
        {
            ret = format_msg_ts_to_json(buf, sizeof(buf), &msg_ts_arr[i]);
            if (ret < 0) {
                uart_out("\n");
                return;
            }
            uart_out(buf);
        }
    }

    // end msg
    uart_out("]}\n");
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

    //LOG_WRN("Got data of length %d", len);

    if (len > sizeof(msg_header) + 2 && !memcmp(msg_header, net_buf_pull_mem(buf, sizeof(msg_header)), sizeof(msg_header))) {
        len -= sizeof(msg_header) + 2; // 2 bytes crc?
        struct msg_ts *rx_msg = net_buf_pull_mem(buf, len);

        // TODO: Use these?
        net_buf_pull_u8(buf);
        net_buf_pull_u8(buf);

        if (len >  0 && len % sizeof (struct msg_ts)  == 0) {
            size_t num_msg = len / sizeof (struct msg_ts);

            // we handle the tx timestamp (the first element)
            {
                k_sem_take(&msg_tx_buf_sem, K_FOREVER);
                struct msg_ts *tx_ts = &rx_msg[0];

                // we push the newest entries to the front (and all others to the back)
                for(int i = NUM_MSG_TS-1; i >= 2 ; i--) {
                    memcpy(&msg_tx_buf[i], &msg_tx_buf[i-1], sizeof(struct msg_ts));
                }
                // we then push the tx ts from the message to the first rx slot
                memcpy(&msg_tx_buf[1], &rx_msg[0], sizeof(struct msg_ts));

                uint64_t rx_ts = dwt_rx_ts(ieee802154_dev);

                msg_tx_buf[1].ts[0] = (uint8_t)(rx_ts&0xFF);
                sys_put_le32(rx_ts >> 8, &msg_tx_buf[1].ts[1]);

                num_receptions++;
                k_sem_give(&msg_tx_buf_sem);
            }
            float cor = dwt_rx_clock_ratio_offset(ieee802154_dev);
            // and simply dump this whole message to the output
            output_msg_to_uart("rx", rx_msg, num_msg, &cor);
        } else {
            LOG_WRN("Got weird data of length %d", len);
            net_pkt_hexdump(pkt, "<");
        }
    } else {
        LOG_WRN("Got WRONG data, pkt %p, len %d", pkt, len);
    }

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
        struct net_pkt *pkt = NULL;
        struct net_buf *buf = NULL;

        size_t len = sizeof (msg_header)+sizeof(msg_tx_buf);

        /* Maximum 2 bytes are added to the len */

        while(pkt == NULL) {
            pkt = net_pkt_alloc_with_buffer(NULL, len, AF_UNSPEC, 0,K_MSEC(100));//K_NO_WAIT);
            if (!pkt) {
                LOG_WRN("COULD NOT ALLOCATE MEMORY FOR PACKET!");
            }
        }

        buf = net_buf_frag_last(pkt->buffer);
        len = net_pkt_get_len(pkt);

        //LOG_DBG("Send pkt %p buf %p len %d", pkt, buf, len);

        //LOG_HEXDUMP_DBG(buf->data, buf->len, "TX Data");

        // transmit the packet
        // TODO: Shall we retry?
        {
            int ret;

            uint64_t uus_delay = 900; // what value to choose here? Depends on the processor etc!
            uint64_t estimated_ts = 0;

            struct net_ptp_time ts;
            ts.second = 0;
            ts.nanosecond = 0;
            net_pkt_set_timestamp(pkt, &ts);

            net_pkt_write(pkt, msg_header, sizeof(msg_header));

            k_sem_take(&msg_tx_buf_sem, K_FOREVER);
            // START OF TIMING SENSITIVE
            {
                estimated_ts = dwt_plan_delayed_tx(ieee802154_dev, uus_delay);

                uint8_t dst[8];
                sys_put_le64(estimated_ts, dst);
                for(int i = 0; i < sizeof(msg_tx_buf[0].ts); i++) {
                    msg_tx_buf[0].ts[i] = dst[i];
                }

                // all other entries are updated in the rx event!
                net_pkt_write(pkt, msg_tx_buf, sizeof(struct msg_ts)*MIN(NUM_MSG_TS, num_receptions+1));

                ret = radio_api->tx(ieee802154_dev, IEEE802154_TX_MODE_TXTIME, pkt, buf);
            }
            // END OF TIMING SENSITIVE

            if (ret) {
                LOG_ERR("TX: Error transmitting data!");
            } else {

                output_msg_to_uart("tx", msg_tx_buf, MIN(NUM_MSG_TS, num_receptions+1), NULL);

                msg_tx_buf[0].sn++;
                sent_packets++;

                uint64_t estimated_ns = dwt_ts_to_fs(estimated_ts) / 1000000U;

                struct net_ptp_time *actual_ts = net_pkt_timestamp(pkt);
                uint64_t actual_ns = actual_ts->second * 1000000000U + actual_ts->nanosecond;

                //LOG_DBG("TX: Estimated %llu Actual %llu", estimated_ns, actual_ns);
            }
            k_sem_give(&msg_tx_buf_sem);
        }

        net_pkt_unref(pkt);

        k_msleep(1000);
    }
}