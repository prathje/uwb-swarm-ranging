#ifndef HISTORY_H
#define HISTORY_H

#include "nodes.h"

#define HISTORY_NUM_ROUNDS (0)
#define HISTORY_LENGTH (HISTORY_NUM_ROUNDS*NUM_NODES)

typedef uint8_t ts_t[5];

struct __attribute__((__packed__)) msg {
    uint16_t round;
    uint8_t number;
    ts_t tx_ts;
    ts_t rx_ts[NUM_NODES]; // we keep the slot for our own nodes (wasting a bit of space in transmissions but making it a lot easier to handle everywhere...)
};

void history_reset();
void history_save(struct msg *msg, int8_t rssi, int8_t applied_bias_correction, int carrierintegrator);
void history_print();

#endif