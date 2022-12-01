#ifndef MESSAGES_H
#define MESSAGES_H

#include <stdint.h>

#include "typedefs.h"

// If we are not using the Zephyr build system (e.g. for unit tests)
#ifndef CONFIG_PAN_ID
#define CONFIG_PAN_ID 0xDECA
#endif

/**
 * @brief Read a timestamp from a message buffer.
 * 
 * @param buffer The point in the buffer where the timestamp is located.
 * @return timestamp_t The timestamp that has been read.
 */
timestamp_t message_read_timestamp(uint8_t* buffer);

/**
 * @brief  Write a timestamp from a message buffer.
 * 
 * @param buffer The point in the buffer where the timestamp is located.
 * @param ts  The timestamp that will be written.
 */
void message_write_timestamp(uint8_t* buffer, timestamp_t ts);

/**
 * @brief Analyses a message buffer.
 * 
 * @param message_buffer A pointer to the byte buffer the message is stored in.
 * @param message_buffer_len The length of the buffer.
 * @param rx_time The time of reception.
 * @return rx_range_info_t The data contained in the message.
 */
rx_range_info_t analyse_message(uint8_t* message_buffer, size_t message_buffer_len, timestamp_t rx_time);

/**
 * @brief Generates a message buffer with the given information
 *
 * @param message_buffer Pointer to the buffer this function will write to
 * @param message_buffer_size Maximum size of the buffer
 * @param received_messages Information about the received messages, contains ID, sequence number and timestamp
 * @param received_messages_len Number of message information
 * @param self Information about the sending antenna
 * @param tx_timestamp The transmission timestamps of this specific message
 * @return size_t The size of the encoded message
 */
size_t construct_message(uint8_t* message_buffer, size_t message_buffer_size, received_message_t* received_messages, size_t received_messages_len, self_t self, timestamp_t tx_timestamp);

#endif
