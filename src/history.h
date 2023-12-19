#ifndef HISTORY_H
#define HISTORY_H

#include "nodes.h"

typedef uint8_t ts_t[5];

void history_reset();
size_t history_count();
int history_save_rx(uint8_t own_number, uint8_t rx_number, uint16_t rx_round, uint16_t rx_slot, uint64_t rx_ts, int32_t carrierintegrator, int8_t rssi, int8_t bias_correction, uint64_t bias_corrected_rx_ts, uint8_t rx_ttcko_rc_phase);
int history_save_tx(uint8_t own_number, uint16_t tx_round, uint16_t tx_slot, uint64_t tx_ts);
void history_print();

#endif