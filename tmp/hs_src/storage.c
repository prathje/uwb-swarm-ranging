#include "typedefs.h"

static timestamp_t tx_timestamps[TX_TIMESTAMP_BLOCKSIZE];

timestamp_t get_tx_timestamp(sequence_number_t sequence_number) {
    return tx_timestamps[sequence_number % TX_TIMESTAMP_BLOCKSIZE];
}

void set_tx_timestamp(sequence_number_t sequence_number, timestamp_t timestamp) {
    tx_timestamps[sequence_number % TX_TIMESTAMP_BLOCKSIZE] = timestamp;
}