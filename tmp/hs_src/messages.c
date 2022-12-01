#include <stdint.h>

#if defined(__ZEPHYR__) // Are we building for Zephyr

#include <logging/log.h>
#include <zephyr.h>

LOG_MODULE_REGISTER(messages);

#else

#include <stdlib.h>

// Dummy definitions for the logs when compiling for POSIX
#define LOG_ERR(a)
#define LOG_WRN(a)
#define LOG_INF(a)
#define LOG_DBG(a)

// Use regular malloc instead of k_malloc when building for POSIX
void* k_malloc(size_t size) {
    return malloc(size);
}

#endif

#include "message_definition.h"
#include "messages.h"
#include "typedefs.h"

timestamp_t message_read_timestamp(uint8_t* buffer) {
    timestamp_t result = 0;
    for (int i = 0; i < TIMESTAMP_SIZE; i++) {
        result <<= 8;
        result |= buffer[i];
    }
    return result;
}

void message_write_timestamp(uint8_t* buffer, timestamp_t ts) {
    for (int i = 0; i < TIMESTAMP_SIZE; i++) {
        buffer[i] = (ts >> (8 * (TIMESTAMP_SIZE - 1 - i))) & 0xFF;
    }
}

rx_range_info_t analyse_message(uint8_t* message_buffer, size_t message_buffer_len, timestamp_t rx_time) {
    rx_range_info_t rx_info;
    rx_info.sender_id = message_buffer[SENDER_ID_IDX_1] | (message_buffer[SENDER_ID_IDX_2] << 8);
    rx_info.sequence_number = message_buffer[SEQUENCE_NUMBER_IDX_1] | (message_buffer[SEQUENCE_NUMBER_IDX_2] << 8);
    rx_info.rx_time = rx_time;
    rx_info.tx_time = message_read_timestamp(message_buffer + TX_TIMESTAMP_IDX);

    rx_info.timestamps_len = (message_buffer_len - RX_TIMESTAMP_OFFSET) / RX_TIMESTAMP_SIZE;
    if (rx_info.timestamps_len > 0) {
        rx_info.timestamps = k_malloc(sizeof(rx_range_timestamp_t) * rx_info.timestamps_len);
    } else {
        rx_info.timestamps = NULL;
    }

    for (size_t i = 0; i < rx_info.timestamps_len; i++) {
        int timestamp_index = RX_TIMESTAMP_OFFSET + i * RX_TIMESTAMP_SIZE;
        rx_info.timestamps[i].node_id = message_buffer[timestamp_index + RX_TIMESTAMP_RANGING_ID_OFFSET] |
                                        (message_buffer[timestamp_index + RX_TIMESTAMP_RANGING_ID_OFFSET + 1] << 8);
        rx_info.timestamps[i].sequence_number =
            message_buffer[timestamp_index + RX_TIMESTAMP_SEQUENCE_NUMBER_OFFSET] |
            (message_buffer[timestamp_index + RX_TIMESTAMP_SEQUENCE_NUMBER_OFFSET + 1] << 8);
        rx_info.timestamps[i].rx_time =
            message_read_timestamp(message_buffer + timestamp_index + RX_TIMESTAMP_TIMESTAMP_OFFSET);
    }
    return rx_info;
}

size_t construct_message(uint8_t* message_buffer,
                         size_t message_buffer_size,
                         received_message_t* received_messages,
                         size_t received_messages_len,
                         self_t self,
                         timestamp_t tx_timestamp) {
    int num_timestamp = 0;
    for (size_t i = 0; i < received_messages_len; i++) {
        if (received_messages[i].sequence_number != 0) {
            num_timestamp++;
        }
    }
    
    size_t message_size = TX_TIMESTAMP_IDX + TIMESTAMP_SIZE;

    message_buffer[FRAME_CONTROL_IDX_1] = 0x88;
    message_buffer[FRAME_CONTROL_IDX_2] = 0x41;
    message_buffer[SEQUENCE_NUMBER_IDX_1] = self.sequence_number & 0xFF;
    message_buffer[SEQUENCE_NUMBER_IDX_2] = (self.sequence_number >> 8) & 0xFF;
    message_buffer[PAN_ID_IDX_1] = CONFIG_PAN_ID & 0xFF;
    message_buffer[PAN_ID_IDX_2] = (CONFIG_PAN_ID >> 8) & 0xFF;
    message_buffer[SENDER_ID_IDX_1] = self.id & 0xFF;
    message_buffer[SENDER_ID_IDX_2] = (self.id >> 8) & 0xFF;

    message_write_timestamp(message_buffer + TX_TIMESTAMP_IDX, tx_timestamp);

    // We use j as an index for the message buffer and i as an index for the `received_messages` list.
    int j = 0;
    for (size_t i = 0; i < received_messages_len; i++) {
        if (received_messages[i].sequence_number != 0) {
            int index = RX_TIMESTAMP_OFFSET + (RX_TIMESTAMP_SIZE * j);
            if (index + RX_TIMESTAMP_SEQUENCE_NUMBER_OFFSET + 1 >= message_buffer_size) {
                LOG_ERR("Message buffer too small");
                break;
            }
            message_buffer[index + RX_TIMESTAMP_RANGING_ID_OFFSET] = received_messages[i].sender_id & 0xFF;
            message_buffer[index + RX_TIMESTAMP_RANGING_ID_OFFSET + 1] = (received_messages[i].sender_id >> 8) & 0xFF;
            message_buffer[index + RX_TIMESTAMP_SEQUENCE_NUMBER_OFFSET] = received_messages[i].sequence_number & 0xFF;
            message_buffer[index + RX_TIMESTAMP_SEQUENCE_NUMBER_OFFSET + 1] =
                (received_messages[i].sequence_number >> 8) & 0xFF;
            message_write_timestamp(message_buffer + index + RX_TIMESTAMP_TIMESTAMP_OFFSET,
                                    received_messages[i].rx_timestamp);
            message_size = index + RX_TIMESTAMP_SIZE;
            j++;
        }
    }
    return message_size;
}
