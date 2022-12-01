#ifndef HOST_INTERFACE_H
#define HOST_INTERFACE_H

#include "typedefs.h"

/**
 * @brief Send the info of an outgoing message to the host.
 * 
 * @param info of the message.
 * @param id of the node.
 */
void process_out_message(tx_range_info_t* info, ranging_id_t id);

/**
 * @brief Send the info of an incoming message to the host.
 * 
 * @param info of the message.
 * @param id of the node.
 */
void process_in_message(rx_range_info_t* info, ranging_id_t id);

#endif
