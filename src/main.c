#include <logging/log.h>
#include <zephyr.h>

#include <net/net_core.h>
#include <net/ieee802154_radio.h>
#include <drivers/ieee802154/dw1000.h>
#include <stdio.h>


#include "nodes.h"
#include "estimation.h"
#include "uart.h"
#include "measurements.h"

LOG_MODULE_REGISTER(main);

#define MAX_PACKETS -1
#define INITIAL_DELAY_MS 5000
#define ROUND_TIMEOUT_MS 25
#define POST_ROUND_DELAY_MS 50
#define ESTIMATION_ROUND_DELAY_MS 5000
#define DEVICE_OFFSET_MULTIPLICATOR 1
#define IS_EST_ROUND(X) ((X)%100 == 0)



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


extern void matrix_test();


static uint16_t own_number = 0;
#define INITIATOR_ID 0
#define IS_INITIATOR (own_number == INITIATOR_ID)


static void output_relative_drifts(uint64_t own_dur[], uint64_t other_dur[]);
static void output_msg_to_uart(struct msg* m);

static int64_t round_start = 0;
static int64_t round_end = 0;
static int64_t last_msg_ms = 0;

// we save the last & current msgs of every node
static struct msg last_msg[NUM_NODES];
static struct msg cur_msg[NUM_NODES];

 //




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

static inline uint64_t get_uint_duration(const ts_t *end_ts, const ts_t *start_ts){

    uint64_t end = ts_to_uint(end_ts);
    uint64_t start = ts_to_uint(start_ts);

    if (end == 0 || start == 0) {
        return 0; // TODO: we might want to change this behavior
    }

    if (end < start) {

        // handle overflow!
        end += 0xFFFFFFFFFF;

        if (end < start) {
            return 0; // it is still wrong...
        }
    }

    return end - start;
}


void on_round_end(uint16_t round_number) {

    // cur_msg contains all messages of the this round
    // last_msg contains all messages of the previous round

    // we cleanup invalid values first
     for(int i = 0; i < NUM_NODES; i++) {
        if(last_msg[i].round != round_number - 1) {
            memset(&last_msg[i], 0, sizeof(last_msg[i]));
            LOG_DBG("LastMsg invalid value of %d, expected %hu, got %hu", i, round_number - 1, last_msg[i].round);
        }
     }

     for(int i = 0; i < NUM_NODES; i++) {
        if(cur_msg[i].round != round_number) {
            memset(&cur_msg[i], 0, sizeof(cur_msg[i]));
            LOG_DBG("CurMsg invalid value of %d, expected %hu, got %hu", i, round_number, cur_msg[i].round);
        }
     }

   // determine all relative drift rates for now:

   static float relative_drifts[NUM_NODES];

   static uint64_t rd_own_dur[NUM_NODES] = {0};
   static uint64_t rd_other_dur[NUM_NODES] = {0};

   for(int i = 0; i < NUM_NODES; i++) {
        if (i == own_number) {
            continue;
        }

        // we can extract last and current timestamps
        rd_other_dur[i] = get_uint_duration(&cur_msg[i].tx_ts, &last_msg[i].tx_ts);

        if (i < own_number) {
            // in this case, the other node transmitted before us
            // so we have to extract rx timestamps from our own cur and last msg
            rd_own_dur[i] = get_uint_duration(&cur_msg[own_number].rx_ts[i], &last_msg[own_number].rx_ts[i]);
        } else {
            // i > own_number: in this case, the rx timestamp from the current round is actually in our current tx_buf and the last one in cur_msg
            rd_own_dur[i] = get_uint_duration(&msg_tx_buf.rx_ts[i], &cur_msg[own_number].rx_ts[i]);
        }

        if ( rd_other_dur[i] != 0 && rd_own_dur[i] != 0) {
            relative_drifts[i] = ((float)rd_own_dur[i]) / (float)(rd_other_dur[i]);
        } else {
            // just set it to zero for now...
            relative_drifts[i] = 0.0;
        }
   }

   // our own relative drift is just one ofc
   rd_other_dur[own_number] = 1;
   rd_own_dur[own_number] = 1;
   relative_drifts[own_number] = 1.0;

    if (IS_EST_ROUND(round_number)) {
        uart_out("{");

        char buf[256];
        snprintf(buf, sizeof(buf), "\"number\": %hhu, \"round\": %hhu", own_number, round_number);
        uart_out(buf);


       uart_out(", \"relative_drifts\": ");
       output_relative_drifts(rd_own_dur, rd_other_dur);

       uart_out(", \"last_msg\": [");

       for(int i = 0; i < NUM_NODES; i++) {
            output_msg_to_uart(&last_msg[i]);

            if(i < NUM_NODES-1) {
                uart_out(", ");
            }
         }
       uart_out("], \"cur_msg\": [");

       for(int i = 0; i < NUM_NODES; i++) {
            output_msg_to_uart(&cur_msg[i]);

            if(i < NUM_NODES-1) {
                uart_out(", ");
            }
        }
       uart_out("]}\n");
    }


    // we now check every combination
    // TODO: we might also just want to check for ourselves
    for (int a = 0; a < NUM_NODES; a++) {
        for(int b = 0; b < NUM_NODES; b++) {
            if(a == b) {
                continue;
            }

            // we extract the ranging as initiated by A:
            uint64_t round_dur_a = 0;
            uint64_t delay_dur_b = 0;

            round_dur_a = get_uint_duration(&cur_msg[a].rx_ts[b], &last_msg[a].tx_ts);

            if (a < b) {
                // the response delay of b should be in the last_msg as well
                delay_dur_b = get_uint_duration(&last_msg[b].tx_ts, &last_msg[b].rx_ts[a]);
            } else {
                // otherwise b transmitted before a, so it should reside in the current round
                // TODO: Since we have a lot of delay between rounds, this round and delay values are too big to be handled with the necessary precision!
                //delay_dur_b = get_uint_duration(&cur_msg[b].tx_ts, &cur_msg[b].rx_ts[a]);
            }

            if (round_dur_a != 0 && delay_dur_b != 0 && relative_drifts[a] != 0.0 && relative_drifts[b] != 0.0) {

                float round_a_corrected = (float)(round_dur_a) * relative_drifts[a];
                float delay_b_corrected =  (float)(delay_dur_b) * relative_drifts[b];

                measurement_t tof_in_uwb_us = (round_a_corrected - delay_b_corrected)*0.5;

                estimation_add_measurement(a, b, tof_in_uwb_us);

                float est_distance_in_m = tof_in_uwb_us*SPEED_OF_LIGHT_M_PER_UWB_TU;

                int64_t est_cm = est_distance_in_m*100;
                //LOG_DBG("Round est cm: %hhu, %hhu, %lld, r:%lld, d: %lld", a, b, est_cm, round_dur_a, delay_dur_b);
            }
        }
    }



    // copy all of the current messages to the last round
    memcpy(&last_msg, &cur_msg, sizeof(last_msg));

    // reset cur_msg!
    memset(&cur_msg, 0, sizeof(cur_msg));

    if (IS_EST_ROUND(round_number)) {
        estimate_all();
    }

}

//
//void on_new_msg(const struct msg *a_new) {
//
//    uint8_t a = a_new->number;
//    struct msg *a_last = &last_msg[a];
//
//    // we first check if the last message we have saved is actually the previous round
//    if (a_new->round == a_last->round + 1) {
//        // we extract all possible combinations now
//        // TODO: We could only do this for a == own_number
//        for(int b = 0; b < NUM_NODES; b++) {
//            if (b == a) {
//                continue;   // ignore packets from the same device
//            }
//
//            struct msg *b_last = &last_msg[b];  // the last message that we have received from B
//
//            if ((a < b && b_last->round == a_last->round) || (a > b && b_last->round == a_new->round)) {   // if a < b then the last message from b should be the same as the a_last, else if a > b then b's round should be the same as A's
//
//                uint64_t init_tx_ts_a, init_rx_ts_b, response_tx_ts_b, response_rx_ts_a;
//
//                init_tx_ts_a = ts_to_uint(&a_last->tx_ts);
//                init_rx_ts_b = ts_to_uint(&b_last->rx_ts[a]);
//
//                response_tx_ts_b = ts_to_uint(&b_last->tx_ts);
//                response_rx_ts_a = ts_to_uint(&a_new->rx_ts[b]);
//
//                if (init_tx_ts_a == 0 || init_rx_ts_b == 0 || response_tx_ts_b == 0 || response_rx_ts_a == 0) {
//                    // there might be a message missing -> we ignore this exchange for now
//                    LOG_DBG("Exchange ignored!");
//                    continue;
//                }
//
//
//                // we need to ensure that our values are sane, i.e., init_tx_ts_a < response_rx_ts_a  and init_rx_ts_b < response_tx_ts_b
//                // if that is the case, an overflow happened wich we need to deal with right now!
//
//                if (init_tx_ts_a >= response_rx_ts_a) {
//                    response_rx_ts_a += 0xFFFFFFFFFF;
//                }
//                if (init_rx_ts_b >= response_tx_ts_b) {
//                    response_tx_ts_b += 0xFFFFFFFFFF;
//                }
//
//                uint64_t round_a = response_rx_ts_a - init_tx_ts_a;
//                uint64_t delay_b = response_tx_ts_b - init_rx_ts_b;
//
//                float round_a_corrected = ((float)(response_rx_ts_a - init_tx_ts_a)) * relative_drifts[a];
//                float delay_b_corrected =  ((float)(response_tx_ts_b - init_rx_ts_b)) * relative_drifts[b];
//
//                if (round_a_corrected != 0.0 && delay_b != 0.0) {
//                    float tof_in_uwb_us = (round_a_corrected - delay_b_corrected);
//
//                    int64_t two_tof = tof_in_uwb_us;
//                    LOG_DBG("Round 2Tof: %hhu, %hhu, %lld", a, b, two_tof);
//                }
//            } else {
//                LOG_DBG("Round Mismatch! %hhu, %hhu, %hu, %hu", a, b, a_new->round, b_last->round);
//            }
//        }
//    }
//
//    memcpy(a_last, a_new, sizeof(struct msg)); // save message in the end
//}



int main(void) {

    LOG_INF("Getting node id");
    int16_t signed_node_id = get_node_number(get_own_node_id());

    if (signed_node_id < 0) {
        LOG_INF("Node number NOT FOUND! Shutting down :(");
        return;
    }

    own_number = signed_node_id;
    LOG_INF("GOT node id: %hhu", own_number);

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
        (void)memset(&last_msg, 0, sizeof(last_msg));
        (void)memset(&cur_msg, 0, sizeof(cur_msg));

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

            int64_t ms_since_last_msg = k_uptime_get() - last_msg_ms;

            if (ms_since_last_msg >= ROUND_TIMEOUT_MS) {

                //LOG_INF("Advancing to new round! (%hu)", msg_tx_buf.round+1);

                k_sem_take(&msg_tx_buf_sem, K_FOREVER); // we take this to be sure that on_round_end finished!
                // we then add more delay!
                if (IS_EST_ROUND(msg_tx_buf.round)) {
                    k_msleep(ESTIMATION_ROUND_DELAY_MS);
                } else {
                    k_msleep(POST_ROUND_DELAY_MS);
                }

                msg_tx_buf.round += 1;
                k_sem_give(&tx_sem);

                k_sem_give(&msg_tx_buf_sem);

                    round_start = k_uptime_get();
                    round_end = 0;
            }
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



static void output_relative_drifts(uint64_t own_dur[], uint64_t other_dur[]) {
    // write open parentheses
    char buf[256];

    uart_out("[");
    for(size_t i = 0; i < NUM_NODES; i ++) {
        snprintf(buf, sizeof(buf), "{\"rx_dur\": %llu, \"tx_dur\": %llu}", own_dur[i], other_dur[i]);
        uart_out(buf);
        if (i < NUM_NODES-1) {
            uart_out(", ");
        }
    }
    uart_out("]");
}


static void output_msg_to_uart(struct msg* m) {

    char buf[256];

    // write open parentheses
    uart_out("{");

    // write round
    snprintf(buf, sizeof(buf), "\"round\": %hu", m->round);
    uart_out(buf);
    uart_out(", ");

    // write number
    snprintf(buf, sizeof(buf), "\"number\": %hhu", m->number);
    uart_out(buf);
    uart_out(", ");

    // write tx ts
    uint64_t ts = ts_to_uint(&m->tx_ts);
    snprintf(buf, sizeof(buf), "\"tx_ts\": %llu", ts);
    uart_out(buf);
    uart_out(", ");


    // write all rx ts:
    uart_out("\"rx_ts: \": [");
    for(size_t i = 0; i < NUM_NODES; i ++) {
        ts = ts_to_uint(&m->rx_ts[i]);
        snprintf(buf, sizeof(buf), "%llu", ts);
        uart_out(buf);

        if (i < NUM_NODES-1) {
            uart_out(", ");
        }
    }
    uart_out("]");

    // end msg
    uart_out("}");
}


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

            uint64_t rx_ts = dwt_rx_ts(ieee802154_dev);
            // save this message for later processing
            memcpy(&cur_msg[rx_number], rx_msg, sizeof(struct msg));

            last_msg_ms = k_uptime_get(); // we save this value to restart the round when we are the initiator

            k_sem_take(&msg_tx_buf_sem, K_FOREVER);
            {
                // save this ts in our tx buf!
                ts_from_uint(&msg_tx_buf.rx_ts[rx_number], rx_ts);

                // we wait for the packet of our predecessor
                if (!IS_INITIATOR && rx_number == msg_tx_buf.number-1) {
                    if (rx_round == msg_tx_buf.round + 1) {
                        // we can advance to the new round without problems
                    } else {
                        // this is problematic!
                        if (rx_round <= msg_tx_buf.round) {
                            LOG_WRN("Received outdated round from predecessor!!!"); // this is NOT good
                            // TODO: how to handle this case?
                        } else {
                            LOG_INF("Values seem outdated... Resetting..."); // note that we can waste time in this case since we are the next to transmit anyway!
                            // the received round is a lot more progressed than we are
                            // we hence cannot be sure that our received timestamps are still valid and reset them!
                            // we reset the tx_buf Note that we still hold msg_tx_buf_sem
                            (void)memset(&msg_tx_buf.rx_ts, 0, sizeof(msg_tx_buf.rx_ts));
                        }
                    }

                    round_start = k_uptime_get();
                    round_end = 0;

                    msg_tx_buf.round = MAX(msg_tx_buf.round, rx_round); // use new updated round number in any case
                    k_sem_give(&tx_sem);

                    //LOG_DBG("Starting new round! (n: %hhu, r: %hu)", msg_tx_buf.number, msg_tx_buf.round);
                } else if (rx_number == NUM_NODES-1) {
                    // oh wow, this was the last one!
                    // we could technically directly start the next round as an initiator
                    int64_t milliseconds_spent = k_uptime_delta(&round_start);
                    LOG_INF("ROUND FINISHED! ms: %lld", milliseconds_spent);
                    on_round_end(msg_tx_buf.round);
                }
            }
            k_sem_give(&msg_tx_buf_sem);

            //num_receptions++;
            //int carrierintegrator = dwt_readcarrierintegrator(ieee802154_dev);
            // and simply dump this whole message to the output
            //output_msg_to_uart("rx", rx_msg, num_msg, &carrierintegrator, &rssi);

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

                if (own_number == NUM_NODES-1) {
                    // oh wow, this was the last one! -> we end the round now
                    int64_t milliseconds_spent = k_uptime_delta(&round_start);
                    LOG_INF("ROUND FINISHED! ms: %lld", milliseconds_spent);
                    on_round_end(msg_tx_buf.round);
                }
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
        last_msg_ms = k_uptime_get();
        transmit();
    }
}