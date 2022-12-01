#include "deca_device_api.h"

#include "typedefs.h"

timestamp_t read_systemtime() {
    // uint32_t nrf_time = (uint32_t) ((double) k_uptime_get() / 1000 / DWT_TIME_UNITS);
    uint8_t timestamp_buffer[5];
    dwt_readsystime(timestamp_buffer);
    timestamp_t timestamp = 0;
    for (int i = 4; i >= 0; i--) {
        timestamp <<= 8;
        timestamp |= timestamp_buffer[i];
    }
    return timestamp;  // | nrf_time << (5 * 8);
}

timestamp_t read_rx_timestamp() {
    // uint32_t nrf_time = (uint32_t) ((double) k_uptime_get() / 1000 / DWT_TIME_UNITS);
    uint8_t timestamp_buffer[5];
    dwt_readrxtimestamp(timestamp_buffer);
    timestamp_t timestamp = 0;
    for (int i = 4; i >= 0; i--) {
        timestamp <<= 8;
        timestamp |= timestamp_buffer[i];
    }
    return timestamp;  // | nrf_time << (5 * 8);
}

timestamp_t read_tx_timestamp() {
    // uint32_t nrf_time = (uint32_t) ((double) k_uptime_get() / 1000 / DWT_TIME_UNITS);
    uint8_t timestamp_buffer[5];
    dwt_readtxtimestamp(timestamp_buffer);
    timestamp_t timestamp = 0;
    for (int i = 4; i >= 0; i--) {
        timestamp <<= 8;
        timestamp |= timestamp_buffer[i];
    }
    return timestamp;  // | nrf_time << (5 * 8);
}
