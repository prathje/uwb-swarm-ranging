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



// we save the last & current msgs of every node
static struct msg last_msg[NUM_NODES];
static struct msg cur_msg[NUM_NODES];

static float relative_drifts[NUM_NODES]; //


static inline void ts_from_uint(ts_t *ts_out, uint64_t val) {
    uint8_t dst[sizeof(uint64_t)];
    sys_put_le64(val, dst);
    memcpy(ts_out, dst, sizeof(ts_t));
}

static inline uint64_t ts_to_uint(const ts_t *ts) {
    uint8_t buf[sizeof(uint64_t)] = {0};
    memcpy(&buf, ts, sizeof(ts_t));
    return sys_get_le64(buf);
}


void on_round_end() {

    // we need to go through all new msgs

    /*


                    uint64_t old_rx_ts = ts_to_uint(&last_msg[own_number].rx_ts[rx_number]);

                ts_from_uint(&msg_tx_buf.rx_ts[rx_number], rx_ts);

                uint64_t old_tx_ts = ts_to_uint(&last_msg[rx_number].tx_ts);
                uint64_t tx_ts = ts_to_uint(&rx_msg->tx_ts);

                if (old_rx_ts != 0) {

                    if (old_rx_ts > rx_ts) {
                        rx_ts += 0xFFFFFFFFFF;
                    }

                    if (old_tx_ts > tx_ts) {
                        tx_ts += 0xFFFFFFFFFF;
                    }

                    if (tx_ts - old_tx_ts != 0) {
                        relative_drifts[rx_number] = (float)(rx_ts - old_rx_ts) / (float)(tx_ts - old_tx_ts);

                    }
                }


                LOG_DBG("Relative drift estimation: %lld, %lld, %lld, %lld", rx_ts, old_rx_ts, tx_ts, old_tx_ts);
                */

}


void on_new_msg(const struct msg *a_new) {

    uint8_t a = a_new->number;
    struct msg *a_last = &last_msg[a];

    // we first check if the last message we have saved is actually the previous round
    if (a_new->round == a_last->round + 1) {
        // we extract all possible combinations now
        // TODO: We could only do this for a == own_number
        for(int b = 0; b < NUM_NODES; b++) {
            if (b == a) {
                continue;   // ignore packets from the same device
            }

            struct msg *b_last = &last_msg[b];  // the last message that we have received from B

            if ((a < b && b_last->round == a_last->round) || (a > b && b_last->round == a_new->round)) {   // if a < b then the last message from b should be the same as the a_last, else if a > b then b's round should be the same as A's

                uint64_t init_tx_ts_a, init_rx_ts_b, response_tx_ts_b, response_rx_ts_a;

                init_tx_ts_a = ts_to_uint(&a_last->tx_ts);
                init_rx_ts_b = ts_to_uint(&b_last->rx_ts[a]);

                response_tx_ts_b = ts_to_uint(&b_last->tx_ts);
                response_rx_ts_a = ts_to_uint(&a_new->rx_ts[b]);

                if (init_tx_ts_a == 0 || init_rx_ts_b == 0 || response_tx_ts_b == 0 || response_rx_ts_a == 0) {
                    // there might be a message missing -> we ignore this exchange for now
                    LOG_DBG("Exchange ignored!");
                    continue;
                }


                // we need to ensure that our values are sane, i.e., init_tx_ts_a < response_rx_ts_a  and init_rx_ts_b < response_tx_ts_b
                // if that is the case, an overflow happened wich we need to deal with right now!

                if (init_tx_ts_a >= response_rx_ts_a) {
                    response_rx_ts_a += 0xFFFFFFFFFF;
                }
                if (init_rx_ts_b >= response_tx_ts_b) {
                    response_tx_ts_b += 0xFFFFFFFFFF;
                }

                uint64_t round_a = response_rx_ts_a - init_tx_ts_a;
                uint64_t delay_b = response_tx_ts_b - init_rx_ts_b;

                float round_a_corrected = ((float)(response_rx_ts_a - init_tx_ts_a)) * relative_drifts[a];
                float delay_b_corrected =  ((float)(response_tx_ts_b - init_rx_ts_b)) * relative_drifts[b];

                if (round_a_corrected != 0.0 && delay_b != 0.0) {
                    float tof_in_uwb_us = (round_a_corrected - delay_b_corrected);

                    int64_t two_tof = tof_in_uwb_us;
                    LOG_DBG("Round 2Tof: %hhu, %hhu, %lld", a, b, two_tof);
                }
            } else {
                LOG_DBG("Round Mismatch! %hhu, %hhu, %hu, %hu", a, b, a_new->round, b_last->round);
            }
        }
    }

    memcpy(a_last, a_new, sizeof(struct msg)); // save message in the end
}



int main(void) {

    LOG_INF("Getting node id");
    own_number = get_node_number(get_own_node_id());

    if (own_number == -1) {
        LOG_INF("Node number NOT FOUND! Shutting down :(");
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

    for(int i = 0; i < NUM_NODES; i++) {
        relative_drifts[i] = 0.0;
    }

    relative_drifts[own_number] = 1.0;

    // prepare msg buffer
    {
        (void)memset(&msg_tx_buf, 0, sizeof(msg_tx_buf));
        (void)memset(&last_msg, 0, sizeof(last_msg));

        // TODO: Add own addr!
        msg_tx_buf.number = own_number&0xFF;
        msg_tx_buf.round = 0;
        (void)memset(&msg_tx_buf.rx_ts, 0, sizeof(msg_tx_buf.rx_ts));
        k_sem_give(&msg_tx_buf_sem);
    }

    /* Setup antenna delay values to 0 to get raw tx values */
    dwt_set_antenna_delay_rx(ieee802154_dev, 16450);
    dwt_set_antenna_delay_tx(ieee802154_dev, 16450);

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

            uint8_t rx_number = rx_msg->number;
            uint16_t rx_round = rx_msg->round;

            // TODO: CHECK IF WE would need to ignore this msg
            // we handle the tx timestamp (the first element)
            {
                // save this message for later processing
                memcpy(&cur_msg[rx_number], rx_msg, sizeof(struct msg));

                k_sem_take(&msg_tx_buf_sem, K_FOREVER);

                uint64_t rx_ts = dwt_rx_ts(ieee802154_dev);

                //LOG_DBG("Received message from %hhu (round %hu)", rx_number, rx_round);

                if (rx_number < msg_tx_buf.number && rx_round > msg_tx_buf.round) {
                    //LOG_DBG("Outdated round detected (round %hu)", msg_tx_buf.round);
                    // we are behind! -> delete timestamps just to be sure? TODO
                    //(void)memset(&msg_tx_buf.rx_ts, 0, sizeof(msg_tx_buf.rx_ts));
                }

                msg_tx_buf.round = MAX(msg_tx_buf.round, rx_round);


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
                    round_start = k_uptime_get();
                    round_end = 0;
                    k_sem_give(&tx_sem);
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

                ts_from_uint(&msg_tx_buf.tx_ts, estimated_ts);

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

                // we also save our own message before resetting it
                memcpy(&cur_msg[own_number], &msg_tx_buf, sizeof(struct msg));

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