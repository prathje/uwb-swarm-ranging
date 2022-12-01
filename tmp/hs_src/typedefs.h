#ifndef TYPEDEFS_H
#define TYPEDEFS_H

#include <stddef.h>
#include <stdint.h>

#define TX_TIMESTAMP_BLOCKSIZE 32

/**
 * @brief Identifier of a node.
 * 
 */
typedef int16_t ranging_id_t;

/**
 * @brief Sequence number of a message.
 * 
 */
typedef int16_t sequence_number_t;

/**
 * @brief Time of a reception or transmission.
 * 
 */
typedef uint64_t timestamp_t;

/**
 * @brief Information about a received message
 * 
 */
typedef struct received_message {
    ranging_id_t sender_id;
    sequence_number_t sequence_number;
    timestamp_t rx_timestamp;
} received_message_t;

/**
 * @brief Information about the node itself.
 * 
 */
typedef struct {
    ranging_id_t id;
    sequence_number_t sequence_number;
} self_t;

/**
 * @brief Timestamp field in `rx_range`
 * 
 */
typedef struct rx_range_timestamp {
    ranging_id_t node_id;
    sequence_number_t sequence_number;
    timestamp_t rx_time;
} rx_range_timestamp_t;

/**
 * @brief Information about a reception event.
 * 
 */
typedef struct rx_range {
    ranging_id_t sender_id;
    sequence_number_t sequence_number;
    timestamp_t tx_time;
    timestamp_t rx_time;
    size_t timestamps_len;
    rx_range_timestamp_t* timestamps;
} rx_range_info_t;

/**
 * @brief Information about a transmission event.
 * 
 */
typedef struct tx_range {
    ranging_id_t id;
    sequence_number_t sequence_number;
    timestamp_t tx_time;
} tx_range_info_t;

#endif