#ifndef TIMESTAMPS_H
#define TIMESTAMPS_H

#include "typedefs.h"

/**
 * @brief Read the systemtime of the DWM module.
 * 
 * @return timestamp_t The systemtime.
 */
timestamp_t read_systemtime();

/**
 * @brief Read the time of the last reception event.
 * 
 * @return timestamp_t The time of the last reception event.
 */
timestamp_t read_rx_timestamp();

/**
 * @brief Read the time of the last transmission event.
 * 
 * @return timestamp_t The time of the last transmission event.
 */
timestamp_t read_tx_timestamp();

#endif