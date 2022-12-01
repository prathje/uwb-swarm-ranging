#include <data/json.h>
#include <drivers/uart.h>
#include <logging/log.h>
#include <stdio.h>
#include <zephyr.h>

#include "host_interface.h"
#include "typedefs.h"

static const struct device* uart_device = DEVICE_DT_GET(DT_CHOSEN(zephyr_console));

void uart_out(char* msg) {
    while (*msg != '\0') {
        uart_poll_out(uart_device, *msg);
        msg++;
    }
}

void process_out_message(tx_range_info_t* info, ranging_id_t id) {
    char buf[90];
    snprintf(buf, 90, "{\"id\":%u,\"tx range\":{\"seq num\":%d,\"tx time\":%llu}}\n", id, info->sequence_number,
             info->tx_time);
    uart_out(buf);
}

void process_in_message(rx_range_info_t* info, ranging_id_t id) {
    char buf[90];
    snprintf(buf, 90, "{\"id\":%u,\"rx range\":{\"sender id\":%d,\"seq num\":%d,", id, info->sender_id,
             info->sequence_number);
    uart_out(buf);
    snprintf(buf, 90, "\"rx time\":%llu,\"tx time\":%llu,\"timestamps\":[", info->rx_time, info->tx_time);
    uart_out(buf);
    for (int i = 0; i < info->timestamps_len; i++) {
        snprintf(buf, 90, "{\"id\":%d,\"seq num\":%d,\"rx time\":%llu}", info->timestamps[i].node_id,
                 info->timestamps[i].sequence_number, info->timestamps[i].rx_time);
        uart_out(buf);
        if (i != info->timestamps_len - 1) {
            uart_out(",");
        }
    }
    uart_out("]}}\n");
}
