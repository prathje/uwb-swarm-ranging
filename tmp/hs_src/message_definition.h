#ifndef MESSAGE_DEFINITION_H
#define MESSAGE_DEFINITION_H

#include "typedefs.h"

/** @brief Lower byte of the frame control. */
#define FRAME_CONTROL_1 0x88
/** @brief Higher byte of the frame control. */
#define FRAME_CONTROL_2 0x41

/**
 * \defgroup message_idx Message indices
 * @{
 */

/**  @brief Index of the lower frame control byte. */
#define FRAME_CONTROL_IDX_1 0
/**  @brief Index of the higher frame control byte. */
#define FRAME_CONTROL_IDX_2 1

/**  @brief Index of the lower pan id byte. */
#define PAN_ID_IDX_1 3
/**  @brief Index of the higher pan id byte. */
#define PAN_ID_IDX_2 4

/**  @brief Index of the lower sequence number byte. */
#define SEQUENCE_NUMBER_IDX_1 5
/**  @brief Index of the higher sequence number byte. */
#define SEQUENCE_NUMBER_IDX_2 6

/**  @brief Index of the lower sender id byte. */
#define SENDER_ID_IDX_1 7
/**  @brief Index of the higher sender id byte. */
#define SENDER_ID_IDX_2 8

/** @brief Index of first by of the transmission timestamp. */
#define TX_TIMESTAMP_IDX 9

/**
 * @}
 */

/** @brief The size of a timestamp field. */
#define TIMESTAMP_SIZE 5

/** @brief The beginning of the receive timestamp section. Index of the first timestamp. */
#define RX_TIMESTAMP_OFFSET (TX_TIMESTAMP_IDX + TIMESTAMP_SIZE)

/**
 * @brief Size of a receive timestamp.
 *
 * @note A receive timestamp must also contain the sequence number and sender id.
 * Those are implicitly given for the transmit timestamp.
 *
 */
#define RX_TIMESTAMP_SIZE (TIMESTAMP_SIZE + sizeof(ranging_id_t) + sizeof(sequence_number_t))

/** @brief The offset of the ranging id within a rx timestamp field. */
#define RX_TIMESTAMP_RANGING_ID_OFFSET 0

/** @brief The offset of the sequence number within a rx timestamp field. */
#define RX_TIMESTAMP_SEQUENCE_NUMBER_OFFSET sizeof(ranging_id_t)

/** @brief The offset of the timestamp within a rx timestamp field. */
#define RX_TIMESTAMP_TIMESTAMP_OFFSET (sizeof(ranging_id_t) + sizeof(sequence_number_t))

#endif
