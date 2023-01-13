#include <logging/log.h>
#include <zephyr.h>

#include <net/net_core.h>
#include <net/ieee802154_radio.h>
#include <drivers/ieee802154/dw1000.h>
#include <stdio.h>
#include <drivers/uart.h>

LOG_MODULE_REGISTER(main);

#define MAX_PACKETS -1
#define INITIAL_DELAY_MS 1000
#define ROUND_TIMEOUT_MS 5000
#define DEVICE_OFFSET_MULTIPLICATOR 1

#define NUM_NODES 14

/* ieee802.15.4 device */
static struct ieee802154_radio_api *radio_api;
static const struct device *ieee802154_dev;




static int sent_packets = 0;

// measurement
// 40 bit measurements
// we save each measurement

// TODO: This is not standard compliant
static uint8_t msg_header[] = {0xDE, 0xCA};

typedef uint8_t ts_t[5];

struct __attribute__((__packed__)) msg {
    uint16_t round;
    uint8_t number;
    ts_t tx_ts;
    ts_t rx_ts[NUM_NODES]; // we keep the slot for our own nodes (wasting a bit of space in transmissions but making it a lot easier to handle everywhere...)
};


static struct msg msg_tx_buf;

K_SEM_DEFINE(tx_sem, 0, 1);
K_SEM_DEFINE(msg_tx_buf_sem, 0, 1);

static void init_tx_queue();


static const struct device* uart_device = DEVICE_DT_GET(DT_CHOSEN(zephyr_console));
static void uart_out(char* msg) {
    while (*msg != '\0') {
        uart_poll_out(uart_device, *msg);
        msg++;
    }
}




extern void matrix_test();


extern uint16_t get_own_node_id();
extern int8_t get_node_number(uint16_t node_id);



static uint16_t own_number = 0;
#define INITIATOR_ID 0
#define IS_INITIATOR (own_number == INITIATOR_ID)




static int64_t round_start = 0;
static int64_t round_end = 0;


int main(void) {

    LOG_INF("Getting node id");
    own_number = get_node_number(get_own_node_id());

    if (own_number == -1) {
        LOG_INF("Node number NOT FOUND!");
        return;
    }

    //LOG_INF("Testing ...");
    //matrix_test();
    //return;


    int ret = 0;
    LOG_INF("Starting ...");

    LOG_INF("Initialize ieee802.15.4");
    ieee802154_dev = device_get_binding(CONFIG_NET_CONFIG_IEEE802154_DEV_NAME);

    if (!ieee802154_dev) {
        LOG_ERR("Cannot get ieee 802.15.4 device");
        return false;
    }

    // prepare msg buffer
    {
        (void)memset(&msg_tx_buf, 0, sizeof(msg_tx_buf));
        // TODO: Add own addr!
        msg_tx_buf.number = own_number&0xFF;
        msg_tx_buf.round = 0;
        (void)memset(&msg_tx_buf.rx_ts, 0, sizeof(msg_tx_buf.rx_ts));
        k_sem_give(&msg_tx_buf_sem);
    }

    /* Setup antenna delay values to 0 to get raw tx values */
    dwt_set_antenna_delay_rx(ieee802154_dev, 0);
    dwt_set_antenna_delay_tx(ieee802154_dev, 0);

    // we disable the frame filter, otherwise the packets are not received!
    dwt_set_frame_filter(ieee802154_dev, 0, 0);

    radio_api = (struct ieee802154_radio_api *)ieee802154_dev->api;

    LOG_INF("Start IEEE 802.15.4 device");
    ret = radio_api->start(ieee802154_dev);
    if(ret) {
        LOG_ERR("Could not start ieee 802.15.4 device");
        return false;
    }

    k_msleep(INITIAL_DELAY_MS);

    // Create TX thread and queue
    init_tx_queue();

    while (1) {

        if (sent_packets > 0 && sent_packets % 1000 == 0) {
            LOG_DBG("Sent %d packets", sent_packets);
        }

        if (IS_INITIATOR) {
            // we advance the round \o/
            msg_tx_buf.round += 1;
            LOG_INF("Advancing new round! (%hu)", msg_tx_buf.round);
            k_sem_give(&tx_sem);

            round_start = k_uptime_get();
            round_end = 0;

            k_msleep(ROUND_TIMEOUT_MS);
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

//
//static int format_msg_ts_to_json(char *buf, size_t buf_len, struct msg_ts *msg_ts) {
//    uint16_t addr = sys_get_le16(&msg_ts->addr[0]);
//    uint64_t ts = ((uint64_t)sys_get_le32(&msg_ts->ts[1]) << 8) | msg_ts->ts[0];
//    return snprintf(buf, buf_len, "{\"addr\": \"0x%04X\", \"sn\": %d, \"ts\": %llu}", addr, msg_ts->sn, ts);
//}
//
//static void output_msg_to_uart(char* type, struct msg_ts msg_ts_arr[], size_t num_ts, int *carrierintegrator, int8_t *rssi) {
//    if (num_ts == 0) {
//        uart_out("{}");
//        return;
//    }
//
//    // write open parentheses
//    uart_out("{ \"type\": \"");
//    uart_out(type);
//    uart_out("\", ");
//
//    char buf[256];
//
//
//    if (carrierintegrator != NULL) {
//       snprintf(buf, sizeof(buf), "\"carrierintegrator\": %d, ", *carrierintegrator);
//        uart_out(buf);
//    }
//
//    if (rssi != NULL) {
//        snprintf(buf, sizeof(buf), "\"rssi\": %d, ", *rssi);
//        uart_out(buf);
//    }
//
//
//    uart_out("\"tx\": ");
//
//
//    int ret = 0;
//    ret = format_msg_ts_to_json(buf, sizeof(buf), &msg_ts_arr[0]);
//
//    if (ret < 0) {
//        uart_out("\n");
//        return;
//    }
//
//    uart_out(buf);
//    uart_out(", \"rx\": [");
//
//    // write message content
//    for(int i = 1; i < num_ts; i++) {
//        // add separators in between
//        if (i > 1)
//        {
//            uart_out(", ");
//        }
//
//        // write ts content
//        {
//            ret = format_msg_ts_to_json(buf, sizeof(buf), &msg_ts_arr[i]);
//            if (ret < 0) {
//                uart_out("\n");
//                return;
//            }
//            uart_out(buf);
//        }
//    }
//
//    // end msg
//    uart_out("]}\n");
//}


enum net_verdict ieee802154_radio_handle_ack(struct net_if *iface, struct net_pkt *pkt)
{
    return NET_CONTINUE;
}

/**
 * Interface to the network stack, will be called when the packet is
 * received
 */
int net_recv_data(struct net_if *iface, struct net_pkt *pkt)
{
    size_t len = net_pkt_get_len(pkt);
    struct net_buf *buf = pkt->buffer;
    int ret = 0;

    int8_t rssi = (int8_t)net_pkt_ieee802154_rssi(pkt);

    //LOG_WRN("Got data of length %d", len);

    if (len > sizeof(msg_header) + 2 && !memcmp(msg_header, net_buf_pull_mem(buf, sizeof(msg_header)), sizeof(msg_header))) {
        len -= sizeof(msg_header) + 2; // 2 bytes crc?
        struct msg *rx_msg = net_buf_pull_mem(buf, len);

        // TODO: Use these?
        net_buf_pull_u8(buf);
        net_buf_pull_u8(buf);

        if (len >  0 && len % sizeof (struct msg)  == 0) {
            //size_t num_msg = len / sizeof (struct msg_ts);

            //TODO: Save the reception somewhere!

            // we handle the tx timestamp (the first element)
            {
                k_sem_take(&msg_tx_buf_sem, K_FOREVER);

                uint64_t rx_ts = dwt_rx_ts(ieee802154_dev);

                uint8_t rx_number = rx_msg->number;
                uint16_t rx_round = rx_msg->round;

                //LOG_DBG("Received message from %hhu (round %hu)", rx_number, rx_round);

                if (rx_number < msg_tx_buf.number && rx_round > msg_tx_buf.round) {
                    //LOG_DBG("Outdated round detected (round %hu)", msg_tx_buf.round);
                    // we are behind! -> delete timestamps just to be sure? TODO
                    //(void)memset(&msg_tx_buf.rx_ts, 0, sizeof(msg_tx_buf.rx_ts));
                }

                msg_tx_buf.round = MAX(msg_tx_buf.round, rx_round);

                // TODO: should this message be ignored?
                    msg_tx_buf.rx_ts[rx_number][0] = (uint8_t)(rx_ts&0xFF);
                    sys_put_le32(rx_ts >> 8, &msg_tx_buf.rx_ts[rx_number][1]);
                    k_sem_give(&msg_tx_buf_sem);

                if ((rx_number < msg_tx_buf.number && rx_round == msg_tx_buf.round +1) || (rx_number > msg_tx_buf.number && rx_round == msg_tx_buf.round)) {

                } else {
                    //LOG_DBG("Message was ignored!");
                }

                bool start_new_round = false;
                if (IS_INITIATOR && rx_number == NUM_NODES-1) {
                    // oh wow, this was the last one!
                    // we could technically directly start the next round
                    // TODO: start the next round
                    //start_new_round = true;
                    int64_t milliseconds_spent = k_uptime_delta(&round_start);
                    LOG_INF("ROUND FINISHED! ms: %lld", milliseconds_spent);
                } else if (!IS_INITIATOR && rx_number == msg_tx_buf.number-1) {
                    // we are not the initiator
                    // we wait for the packet of our predecessor
                    start_new_round = true;
                }

                if (start_new_round) {
                    //LOG_DBG("Starting new round! (n: %hhu, r: %hu)", msg_tx_buf.number, msg_tx_buf.round);
                    k_sem_give(&tx_sem);
                    round_start = k_uptime_get();
                    round_end = 0;
                }

                k_sem_give(&msg_tx_buf_sem);

                //num_receptions++;

                //int carrierintegrator = dwt_readcarrierintegrator(ieee802154_dev);
                // and simply dump this whole message to the output
                //output_msg_to_uart("rx", rx_msg, num_msg, &carrierintegrator, &rssi);


            }

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




static int transmit() {

        struct net_pkt *pkt = NULL;
        struct net_buf *buf = NULL;

        size_t len = sizeof (msg_header)+sizeof(msg_tx_buf);

        /* Maximum 2 bytes are added to the len */

        while(pkt == NULL) {
            pkt = net_pkt_alloc_with_buffer(NULL, len, AF_UNSPEC, 0, K_MSEC(100));//K_NO_WAIT);
            if (!pkt) {
                LOG_WRN("COULD NOT ALLOCATE MEMORY FOR PACKET!");
            }
        }

        buf = net_buf_frag_last(pkt->buffer);
        len = net_pkt_get_len(pkt);

        //LOG_DBG("Send pkt %p buf %p len %d", pkt, buf, len);

        //LOG_HEXDUMP_DBG(buf->data, buf->len, "TX Data");

        // transmit the packet
        int ret;
        {

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

                // put planned ts into the packet!
                uint8_t dst[8];
                sys_put_le64(estimated_ts, dst);
                memcpy(&msg_tx_buf.tx_ts, dst, sizeof(msg_tx_buf.tx_ts));

                // all other entries are updated in the rx event!
                net_pkt_write(pkt, &msg_tx_buf, sizeof(msg_tx_buf));
                ret = radio_api->tx(ieee802154_dev, IEEE802154_TX_MODE_TXTIME, pkt, buf);
            }
            // END OF TIMING SENSITIVE

            if (ret) {
                LOG_ERR("TX: Error transmitting data!");
            } else {

                //output_msg_to_uart("tx", msg_tx_buf, MIN(NUM_MSG_TS, num_receptions+1), NULL, NULL);

                //msg_tx_buf[0].sn++;
                //sent_packets++;

                //uint64_t estimated_ns = dwt_ts_to_fs(estimated_ts) / 1000000U;
                //struct net_ptp_time *actual_ts = net_pkt_timestamp(pkt);
                //uint64_t actual_ns = actual_ts->second * 1000000000U + actual_ts->nanosecond;
                //LOG_DBG("TX: Estimated %llu Actual %llu", estimated_ns, actual_ns);

                // we reset the tx_buf Note that we still hold msg_tx_buf_sem
                (void)memset(&msg_tx_buf.rx_ts, 0, sizeof(msg_tx_buf.rx_ts));
            }
            k_sem_give(&msg_tx_buf_sem);
        }

        net_pkt_unref(pkt);
        return ret;
}

/**
 * Stack for the tx thread.
 */
static K_THREAD_STACK_DEFINE(tx_stack, 2048);
static struct k_thread tx_thread_data;
static void tx_thread(void);
static struct k_fifo tx_queue;

#define TX_THREAD_PRIORITY K_PRIO_COOP(CONFIG_NUM_COOP_PRIORITIES - 1)
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
        k_sem_take(&tx_sem, K_FOREVER);
        transmit();
    }
}