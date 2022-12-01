#ifndef STORAGE_H
#define STORAGE_H

#include "typedefs.h"

/**
 * @brief Get a tx timestamp a tx timestamp from the storage
 *  
 * @param sequence_number of the timestamp we need.
 * @return timestamp_t the stored timestamp.
 */
timestamp_t get_tx_timestamp(sequence_number_t sequence_number);

/**
 * @brief Save a tx timestamp
 * 
 * @param sequence_number of the timestamp we want to store.
 * @param timestamp the timestamp to be stored.
 */
void set_tx_timestamp(sequence_number_t sequence_number, timestamp_t timestamp);

#endif