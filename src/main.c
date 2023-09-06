#include <logging/log.h>
#include <zephyr.h>

#include <net/net_core.h>
#include <net/ieee802154_radio.h>
#include <drivers/ieee802154/dw1000.h>
#include <stdio.h>


#include "nodes.h"
#include "estimation.h"
#include "log.h"
#include "measurements.h"
#include "history.h"

LOG_MODULE_REGISTER(main);

#define LOG_FLUSH_DIRECTLY 0

#define NUM_ROUNDS (10)

#define INITIAL_DELAY_MS 5000
#define SLOT_DUR_UUS 2000
#define TX_BUFFER_DELAY_UUS 1000
#define TX_INVOKE_MIN_DELAY_UUS 500
#define POST_ROUND_DELAY_UUS 10000000
#define PRE_ROUND_DELAY_UUS 5000000
#define DWT_TS_MASK (0xFFFFFFFFFF)

// Debug values
//#define SLOT_DUR_UUS 2000000
//#define TX_BUFFER_DELAY_UUS 500000


#define LOG_SCHEDULING 0

#define ROUND_DUR_US (NUM_SLOTS*SLOT_DUR_US+POST_ROUND_DELAY_US)


#define INITIAL_DELAY_US (5000000)

#if LOG_FLUSH_DIRECTLY
#define log_out uart_out
#else
#define log_out log_out
#endif

#define FINAL_ESTIMATION_TIMEOUT_MS (ROUND_TIMEOUT_MS*10)

/* ieee802.15.4 device */
static struct ieee802154_radio_api *radio_api;
static const struct device *ieee802154_dev;

// We keep track of time in terms of dwt timestamps (which in theory should be pretty good ignoring the relative clock drift)
static uint64_t round_start_dwt_ts = 0;
static uint64_t last_round_start_dwt_ts = 0;

// measurement
// 40 bit measurements
// we save each measurement

// TODO: This is not standard compliant
static uint8_t msg_header[] = {0xDE, 0xCA};

static uint16_t own_number = 0;
static uint32_t cur_round = 0;
static uint32_t next_slot = 0;

// As we are logging everything, we do not need to actually send timestamps here.
struct __attribute__((__packed__)) msg {
    uint8_t number; // the tx sender number
    uint32_t round; // The round is always started by the first node
    uint32_t slot; // The current slot id, there are slots equal to the number of PAIRS in the system, i.e. n squared
};

static struct msg msg_tx_buf;

// Rounds are always started by the first node from which point on all other slots are being synchronized.


#define INITIATOR_ID 0
#define IS_INITIATOR (own_number == INITIATOR_ID)


#define SLOT_IDLE -1
#define SLOT_LOG_FLUSH -2


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


#define DWT_TS_TO_US(X) (((X)*15650)/1000000000)
#define UUS_TO_DWT_TS(X) (((uint64_t)X)*(uint64_t)65536)

// this function blocks until the wanted ts is almost reached, i.e. it determines the number of us between the wanted and the current ts and blocks for that duration (also adding a correction value)
// Note that it is not precise since we rely on the CPU cycles, yet, we just use it for rough syncs with the dwt clock
// A small correction factor for the whole function execution

// This function execution on its own has an overhead of roughly 83 us. We add a bit of buffer time to be kind of sure to schedule stuff correctly
#define DWT_BUSY_WAIT_DWT_CORRECTION_US (100)
static void busy_wait_until_dwt_ts(uint64_t wanted_ts) {
    uint64_t init_ts = dwt_system_ts(ieee802154_dev);
    uint64_t diff = DWT_TS_TO_US(((uint64_t)(wanted_ts-init_ts))&DWT_TS_MASK); // This should wrap around nicely

    // We check that our wait call might not delay us too much
    if (diff > DWT_BUSY_WAIT_DWT_CORRECTION_US && diff < DWT_TS_TO_US(DWT_TS_MASK/2)) {
        k_busy_wait(diff - DWT_BUSY_WAIT_DWT_CORRECTION_US);
    }
}

// This function execution on its own has an overhead of roughly 83 us. We add a bit of buffer time to be kind of sure to schedule stuff correctly
#define DWT_SLEEP_DWT_CORRECTION_US (100+50)
static void sleep_until_dwt_ts(uint64_t wanted_ts) {
    uint64_t init_ts = dwt_system_ts(ieee802154_dev);
    uint64_t diff = DWT_TS_TO_US(((uint64_t)(wanted_ts-init_ts))&DWT_TS_MASK); // This should wrap around nicely

    // We check that our sleep call might not delay us too much
    if (diff > DWT_SLEEP_DWT_CORRECTION_US && diff < DWT_TS_TO_US(DWT_TS_MASK/2)) {
        k_usleep(diff - DWT_SLEEP_DWT_CORRECTION_US);
    }
}


K_SEM_DEFINE(round_start_sem, 0, 1);

uint64_t schedule_get_slot_duration_dwt_ts(uint16_t r, uint16_t slot) {
    return UUS_TO_DWT_TS(SLOT_DUR_UUS);
}

int8_t schedule_get_tx_node_number(uint32_t r, uint32_t slot) {

    uint16_t exchange = slot / 3;
    uint8_t m = slot % 3;

    uint16_t init = exchange / (NUM_NODES - 1);
    uint16_t resp = exchange % (NUM_NODES - 1);

    if(init == resp) {
        resp = (resp+1) % NUM_NODES; // we do not want to execute a ranging with ourselves..., actually modulo should not be necessary here anyway?
    }

    if (m == 0 || m == 2) {
        return init;
    } else if(m == 1) {
        return resp;
    }

    return -1;
}


int main(void) {


    LOG_INF("Getting node id");
    int16_t signed_node_id = get_node_number(get_own_node_id());

    if (signed_node_id < 0) {
        LOG_INF("Node number NOT FOUND! Shutting down :( I am: 0x%04hx", get_own_node_id());
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
        msg_tx_buf.number = own_number&0xFF;
        msg_tx_buf.round = 0;
    }

    /* Setup antenna delay values to 0 to get raw tx values */
    uint32_t opt_delay_both = dwt_otp_antenna_delay(ieee802154_dev);
    uint16_t rx_delay = opt_delay_both&0xFFFF; // we fully split the delay
    uint16_t tx_delay = opt_delay_both&0xFFFF; // we fully split the delay

    {
       char buf[512];
       snprintf(buf, sizeof(buf), "{\"event\": \"init\", \"own_number\": %hhu, \"rx_delay\": %hu, \"tx_delay\": %hu}\n", own_number, rx_delay, tx_delay);
       log_out(buf); // this will be flushed later!
       log_flush();
    }


    //dwt_set_antenna_delay_rx(ieee802154_dev, 16450);
    //dwt_set_antenna_delay_tx(ieee802154_dev, 16450);

    // we disable the frame filter, otherwise the packets are not received!
    dwt_set_frame_filter(ieee802154_dev, 0, 0);

    radio_api = (struct ieee802154_radio_api *)ieee802154_dev->api;

    LOG_INF("Start IEEE 802.15.4 device");
    ret = radio_api->start(ieee802154_dev);

    if(ret) {
        LOG_ERR("Could not start ieee 802.15.4 device");
        return false;
    }

    // Sleep in DWT time to have enough time before the round starts.
    sleep_until_dwt_ts(dwt_system_ts(ieee802154_dev)+UUS_TO_DWT_TS(INITIAL_DELAY_MS*1000) & DWT_TS_MASK);

    {
        uint64_t init_ts = dwt_system_ts(ieee802154_dev);
        uint64_t wanted_ts = init_ts + UUS_TO_DWT_TS(100);
        sleep_until_dwt_ts(wanted_ts);
        uint64_t other_ts = dwt_system_ts(ieee802154_dev);

        int64_t diff = (int64_t)other_ts - (int64_t)wanted_ts;
        int64_t diff_us = DWT_TS_TO_US(diff);
        LOG_INF("Blocking DWT TS initial  %llu, wanted: %llu, actual: %llu, diff %lld, diff us %lld", init_ts, wanted_ts, other_ts, diff, diff_us);

        init_ts = dwt_system_ts(ieee802154_dev);
        wanted_ts = init_ts + UUS_TO_DWT_TS(TX_BUFFER_DELAY_UUS);
        sleep_until_dwt_ts(wanted_ts);
        other_ts = dwt_system_ts(ieee802154_dev);

        diff = (int64_t)other_ts - (int64_t)wanted_ts;
        diff_us = DWT_TS_TO_US(diff);
        LOG_INF("SLEEPING DWT TS initial  %llu, wanted: %llu, actual: %llu, diff %lld, diff us %lld", init_ts, wanted_ts, other_ts, diff, diff_us);
    }


    uint16_t antenna_delay = dwt_antenna_delay_tx(ieee802154_dev);

    while(cur_round < NUM_ROUNDS) {
        //LOG_INF("Starting round!");

        uint64_t actual_round_start = dwt_system_ts(ieee802154_dev);

        if (own_number == schedule_get_tx_node_number(cur_round, next_slot)) {
            round_start_dwt_ts = (dwt_system_ts(ieee802154_dev)+UUS_TO_DWT_TS((uint64_t)TX_BUFFER_DELAY_UUS)+UUS_TO_DWT_TS(PRE_ROUND_DELAY_UUS)) & DWT_TS_MASK; // we are the first to transmit in this round!

            k_sem_give(&round_start_sem);
        }

        k_sem_take(&round_start_sem, K_FOREVER);

        if (LOG_SCHEDULING && TX_BUFFER_DELAY_UUS >= 2000) {
            char buf[512];
            snprintf(buf, sizeof(buf), "{\"event\": \"round_start\", \"own_number\": %hhu, \"round\": %u, \"round_start\": %llu, \"round_start_us\": %llu, \"cur_us\": %llu}", own_number, cur_round, round_start_dwt_ts, DWT_TS_TO_US(round_start_dwt_ts), DWT_TS_TO_US(dwt_system_ts(ieee802154_dev)));
            log_out(buf);
        }

        uint64_t next_slot_tx_ts = round_start_dwt_ts;

        if (next_slot > 0) {
            // seems like we start not on the first slot (happens when we are not initializing the round!)
            next_slot_tx_ts = (next_slot_tx_ts + schedule_get_slot_duration_dwt_ts(cur_round, next_slot-1)) & DWT_TS_MASK;
        }

        // round_start should be now set!
        while(next_slot < NUM_SLOTS) {

            uint64_t next_slot_dur_ts = schedule_get_slot_duration_dwt_ts(cur_round, next_slot);
            int16_t slot_tx_id = schedule_get_tx_node_number(cur_round, next_slot);

            if (own_number == slot_tx_id) {
                //k_thread_priority_set(k_current_get(), K_PRIO_COOP(CONFIG_NUM_COOP_PRIORITIES - 1)); // we are a bit time sensitive from here on now ;)
                k_thread_priority_set(k_current_get(), K_HIGHEST_THREAD_PRIO); // we are a bit time sensitive from here on now ;)

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

                struct net_ptp_time ts;
                ts.second = 0;
                ts.nanosecond = 0;

                net_pkt_set_timestamp(pkt, &ts);

                net_pkt_write(pkt, msg_header, sizeof(msg_header));

                // update current values
                msg_tx_buf.round = cur_round;
                msg_tx_buf.slot = next_slot;


                // all other entries are updated in the rx event!
                net_pkt_write(pkt, &msg_tx_buf, sizeof(msg_tx_buf));

                uint32_t planned_tx_short_ts = next_slot_tx_ts >> 8;
                dwt_set_delayed_tx_short_ts(ieee802154_dev, planned_tx_short_ts);

                uint64_t tx_invoke_ts = dwt_system_ts(ieee802154_dev);

                // Check that we are not overflowing, i.e. that we are not too late with the tx invocation here! (otherwise we will get a warning which destroys all of our timing...)
                if ((uint64_t)(next_slot_tx_ts-(tx_invoke_ts+DWT_TS_TO_US(TX_INVOKE_MIN_DELAY_UUS))) < DWT_TS_MASK/2) {
                    ret = radio_api->tx(ieee802154_dev, IEEE802154_TX_MODE_TXTIME, pkt, buf);
                }

                // WE NEED COOP PRIORITY otherwise we are verryb likely to miss our tx window
                k_thread_priority_set(k_current_get(), K_HIGHEST_APPLICATION_THREAD_PRIO); // we are less time sensitive from here on now ;)
                net_pkt_unref(pkt);

                if (history_save_tx(own_number, cur_round, next_slot, next_slot_dur_ts)) {
                    LOG_WRN("Could not save TX to history");
                }

                if (LOG_SCHEDULING && DWT_TS_TO_US(next_slot_dur_ts) >= 2000) {
                    char buf[512];
                    snprintf(buf, sizeof(buf), "{\"event\": \"slot_start\", \"own_number\": %hhu, \"round\": %u, \"slot\": %u, \"slot_start\": %llu, \"slot_start_us\": %llu, \"tx_invoke_ts_us\": %llu, \"actual_round_start_us\": %llu}\n", own_number, cur_round, next_slot, next_slot_tx_ts, DWT_TS_TO_US(next_slot_tx_ts), DWT_TS_TO_US(tx_invoke_ts), DWT_TS_TO_US(actual_round_start));
                    log_out(buf);
                }

            } else {
                // TODO: some debugging stuff!
//                if (history_save_rx(own_number, 125, cur_round, next_slot, 1, 2, 3, 4, 5)) {
//                    LOG_WRN("Could not save RX to history");
//                }


                // TODO: we could do stuff here but whatever ;)
                // we should receive packets though, let's see how

                if (LOG_SCHEDULING && DWT_TS_TO_US(next_slot_dur_ts) >= 2000000) {
                    log_flush();
                }

                sleep_until_dwt_ts(((uint64_t)next_slot_tx_ts+(uint64_t)next_slot_dur_ts-(uint64_t)UUS_TO_DWT_TS(TX_BUFFER_DELAY_UUS))& DWT_TS_MASK);
                // TODO: Maybe we can print some log output here?
            }

            // we are already in the next slot, set the next slot tx timestamp accordingly
            next_slot_tx_ts = (next_slot_tx_ts + next_slot_dur_ts) & DWT_TS_MASK;
            next_slot++;
        }

        if (LOG_SCHEDULING) {
            char buf[512];
            // TODO: it is quite possible that this logging breaks the start of the round already!!!
            snprintf(buf, sizeof(buf), "{\"event\": \"round_end\", \"own_number\": %hhu, \"round\": %u, \"round_start_us\": %llu, \"cur_us\": %llu, \"actual_round_start_us\": %llu}\n", own_number, cur_round, DWT_TS_TO_US(round_start_dwt_ts), DWT_TS_TO_US(dwt_system_ts(ieee802154_dev)), DWT_TS_TO_US(actual_round_start));
            log_out(buf);
        }

        last_round_start_dwt_ts = round_start_dwt_ts;
        next_slot = 0; // we restart the round
        round_start_dwt_ts = 0;
        cur_round++;

        // After every round, we flush all of our logs
        uint64_t before_flush_us = dwt_system_ts(ieee802154_dev);
        size_t log_count = history_count();

        history_print();
        history_reset();

        LOG_INF("Flushing before us, after us: %llu, %llu, count %d", DWT_TS_TO_US(before_flush_us), DWT_TS_TO_US(dwt_system_ts(ieee802154_dev)), log_count);

        sleep_until_dwt_ts(before_flush_us + UUS_TO_DWT_TS(POST_ROUND_DELAY_UUS));
    }

    return 0;
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

    //LOG_WRN("Got data of length %d", len);

    if (len > sizeof(msg_header) + 2 && !memcmp(msg_header, net_buf_pull_mem(buf, sizeof(msg_header)), sizeof(msg_header))) {
        len -= sizeof(msg_header) + 2; // 2 bytes crc?
        struct msg *rx_msg = net_buf_pull_mem(buf, len);

        // TODO: Use these?
        net_buf_pull_u8(buf);
        net_buf_pull_u8(buf);

        if (len >  0 && len % sizeof (struct msg)  == 0) {
            //size_t num_msg = len / sizeof (struct msg_ts);

            // TODO: Check that rx_number is actually valid... -> buffer overflow!
            // TODO: CHECK IF WE would need to ignore this msg
            uint8_t rx_number = rx_msg->number;
            uint16_t rx_round = rx_msg->round;
            uint16_t rx_slot = rx_msg->slot;

            //LOG_DBG("Received (n: %hhu, r: %hu)", rx_number, rx_round);

            uint64_t rx_ts = dwt_rx_ts(ieee802154_dev);
            int carrierintegrator = dwt_readcarrierintegrator(ieee802154_dev);
            int8_t rssi = (int8_t)net_pkt_ieee802154_rssi(pkt);
            int8_t bias_correction = get_range_bias_by_rssi(rssi);
            uint64_t bias_corrected_rx_ts = rx_ts - bias_correction;

            // Log the message!
            {
                if (history_save_rx(own_number, rx_number, rx_round, rx_slot, rx_ts, carrierintegrator, rssi, bias_correction, bias_corrected_rx_ts)) {
                    LOG_WRN("Could not save RX to history");
                }
            }

            if (next_slot == 0) {
                cur_round = MAX(cur_round, rx_round); // we might have missed a round!
                next_slot = rx_slot+1; // this should always be > 0 so we do not execute this thing twice...
                round_start_dwt_ts = rx_ts; // TODO: we neglect any airtime and other delays at this point but it should be "good enough"
                k_sem_give(&round_start_sem);
            }

            //LOG_INF("RX Event");

        } else {
            LOG_WRN("Got weird data of length %d", len);
        }
    } else {
        LOG_WRN("Got WRONG data, pkt %p, len %d", pkt, len);
    }

    net_pkt_unref(pkt);

    return ret;
}