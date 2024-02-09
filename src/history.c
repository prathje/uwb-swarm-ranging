#include <logging/log.h>
#include <zephyr.h>

#include <net/net_core.h>
#include <net/ieee802154_radio.h>
#include <drivers/ieee802154/dw1000.h>
#include <stdio.h>

#include "history.h"
#include "log.h"

#define HISTORY_LENGTH 1420

K_SEM_DEFINE(hist_buf_sem, 1, 1);

struct __attribute__((__packed__)) history_record {
    uint8_t txrx_number; // of rx_number == own_number it was a tx!
    uint8_t own_number;
    int8_t rssi;
    int8_t bias_correction;
    uint16_t txrx_round;
    uint16_t txrx_slot;
    uint64_t txrx_ts;
    uint64_t bias_corrected_rx_ts;
    int32_t carrierintegrator;
    //uint8_t rx_ttcko_rc_phase;
};

static int num_stored = 0;

static struct history_record history[HISTORY_LENGTH] = {0};

void history_reset() {
    k_sem_take(&hist_buf_sem, K_FOREVER);
    memset(&history, 0, sizeof(history));
    num_stored = 0;
    k_sem_give(&hist_buf_sem);
}

size_t history_count() {
    return num_stored;
}

int history_save_rx(uint8_t own_number, uint8_t rx_number, uint16_t rx_round, uint16_t rx_slot, uint64_t rx_ts, int32_t carrierintegrator, int8_t rssi, int8_t bias_correction, uint64_t bias_corrected_rx_ts, uint8_t rx_ttcko_rc_phase) {
    k_sem_take(&hist_buf_sem, K_FOREVER);

    if (num_stored == HISTORY_LENGTH){
        k_sem_give(&hist_buf_sem);
        return -1;
    }

    struct history_record *h = &history[num_stored];

    h->txrx_number = rx_number; // of rx_number == own_number it was a tx!
    h->own_number = own_number;
    h->rssi = rssi;
    h->bias_correction = bias_correction;
    h->txrx_round = rx_round;
    h->txrx_slot = rx_slot;
    h->txrx_ts = rx_ts;
    h->bias_corrected_rx_ts = bias_corrected_rx_ts;
    h->carrierintegrator = carrierintegrator;
    //h->rx_ttcko_rc_phase = rx_ttcko_rc_phase;

    num_stored++;

    k_sem_give(&hist_buf_sem);
    return 0;
}

int history_save_tx(uint8_t own_number, uint16_t tx_round, uint16_t tx_slot, uint64_t tx_ts) {
    k_sem_take(&hist_buf_sem, K_FOREVER);

    if (num_stored == HISTORY_LENGTH){
        k_sem_give(&hist_buf_sem);
        return -1;
    }

    struct history_record *h = &history[num_stored];

    h->txrx_number = own_number; // of rx_number == own_number it was a tx!
    h->own_number = own_number;
    h->txrx_round = tx_round;
    h->txrx_slot = tx_slot;
    h->txrx_ts = tx_ts;
    h->rssi = 0;
    h->bias_correction = 0;
    h->bias_corrected_rx_ts = 0;
    h->carrierintegrator = 0;
    //h->rx_ttcko_rc_phase = 0;

    num_stored++;

    k_sem_give(&hist_buf_sem);
    return 0;
}


void history_print() {

    k_sem_take(&hist_buf_sem, K_FOREVER);

    char buf[768];
    for (size_t i = 0; i < num_stored; i++) {

        struct history_record *h = &history[i];

        if (h->own_number != h->txrx_number) {
            //snprintf(buf, sizeof(buf), "{\"event\": \"rx\", \"own_number\": %hhu, \"rx_number\": %hhu, \"rx_round\": %hu, \"rx_slot\": %hu, \"rx_ts\": %llu, \"ci\": %d, \"rssi\": %hhd, \"bias_correction\": %hhd, \"bias_corrected_rx_ts\": %llu, \"rx_ttcko_rc_phase\": %hhu}\n", h->own_number, h->txrx_number, h->txrx_round, h->txrx_slot, h->txrx_ts, h->carrierintegrator, h->rssi, h->bias_correction, h->bias_corrected_rx_ts, h->rx_ttcko_rc_phase);
            snprintf(buf, sizeof(buf), "{\"event\": \"rx\", \"own_number\": %hhu, \"rx_number\": %hhu, \"rx_round\": %hu, \"rx_slot\": %hu, \"rx_ts\": %llu, \"ci\": %d, \"rssi\": %hhd, \"bias_correction\": %hhd, \"bias_corrected_rx_ts\": %llu}\n", h->own_number, h->txrx_number, h->txrx_round, h->txrx_slot, h->txrx_ts, h->carrierintegrator, h->rssi, h->bias_correction, h->bias_corrected_rx_ts);
        } else {
            // this is a transmission.
            snprintf(buf, sizeof(buf), "{\"event\": \"tx\", \"own_number\": %hhu, \"tx_round\": %hu, \"tx_slot\": %hu, \"tx_ts\": %llu}\n", h->own_number, h->txrx_round, h->txrx_slot, h->txrx_ts);
        }
        uart_out(buf);
    }
    k_sem_give(&hist_buf_sem);
}