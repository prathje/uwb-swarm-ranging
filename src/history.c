#include <logging/log.h>
#include <zephyr.h>

#include <net/net_core.h>
#include <net/ieee802154_radio.h>
#include <drivers/ieee802154/dw1000.h>
#include <stdio.h>

#include "history.h"
#include "uart.h"

#if HISTORY_LENGTH > 0

struct __attribute__((__packed__)) history_record {
    struct msg msg;
    int8_t rssi;
    int8_t applied_bias_correction;
    int carrierintegrator;
};

static int num_stored = 0;

static struct history_record history[HISTORY_NUM_ROUNDS*NUM_NODES];

void history_reset() {
    memset(&history, 0, sizeof(history));
    num_stored = 0;
}

void history_save(struct msg *msg, int8_t rssi, int8_t applied_bias_correction, int carrierintegrator) {

    if (num_stored == HISTORY_LENGTH -1){
        return;
    }

    num_stored += 1;

    struct history_record *h = &history[num_stored];
    h->msg = *msg; //copy everything!
    h->rssi = rssi;
    h->applied_bias_correction = applied_bias_correction;
    h->carrierintegrator = carrierintegrator;
}

static inline uint64_t ts_to_uint(const ts_t *ts) {
    uint8_t buf[sizeof(uint64_t)] = {0};
    memcpy(&buf, ts, sizeof(ts_t));
    return sys_get_le64(buf);
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

static void output_record_to_uart(struct history_record* m) {

    char buf[256];

    // write open parentheses
    uart_out("{");

    uart_out("\"msg\":");
    output_msg_to_uart(&m->msg);
    uart_out(", ");

    // write rssi
    snprintf(buf, sizeof(buf), "\"rssi\": %hhd", m->rssi);
    uart_out(buf);
    uart_out(", ");

    // write applied_bias_correction
    snprintf(buf, sizeof(buf), "\"bias\": %hhd", m->applied_bias_correction);
    uart_out(buf);
    uart_out(", ");

    // write carrierintegrator
    snprintf(buf, sizeof(buf), "\"ci\": %d", m->carrierintegrator);
    uart_out(buf);

    // end msg
    uart_out("}");
}


void history_print_entries() {

    // we simply dump all recorded records!
     char buf[256];

    // write open parentheses
    uart_out("[");

    for(int i = 0; i < num_stored; i++) {
        output_record_to_uart(&history[i]);
           if (i < num_stored-1) {
            uart_out(", ");
        }
    }

    // write close parentheses
    uart_out("]");
}

void history_print() {
    uart_out("{\"type\": \"history\", \"entries\": ");
    history_print_entries();
    uart_out("}");
}

#else
// no history is saved! -> NOOP!

void history_reset();
void history_save(struct msg *msg, int8_t rssi, int8_t applied_bias_correction, int carrierintegrator) {}
void history_print() {}

#endif