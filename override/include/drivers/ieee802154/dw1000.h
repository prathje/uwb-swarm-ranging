/*
 * Copyright (c) 2017 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ZEPHYR_INCLUDE_DRIVERS_IEEE802154_DW1000_H_
#define ZEPHYR_INCLUDE_DRIVERS_IEEE802154_DW1000_H_

#include <device.h>

/**
 * Set the upper 32 bits of the dwt timestamp
 * This method should be called just before the invocation of the tx method and from within the same thread
 */
void dwt_set_delayed_tx_short_ts(const struct device *dev, uint32_t short_ts);

/**
 * Sets the delayed tx and returns the estimated tx ts
 * @param dev The dw1000 device
 * @param uus_delay the delay in uwb microseconds
 * @return estimated tx ts (corrected by antenna delay)
 */
uint64_t dwt_plan_delayed_tx(const struct device *dev, uint64_t uus_delay);

void dwt_set_antenna_delay_rx(const struct device *dev, uint16_t rx_delay_ts);
void dwt_set_antenna_delay_tx(const struct device *dev, uint16_t tx_delay_ts);

uint16_t dwt_antenna_delay_rx(const struct device *dev);
uint16_t dwt_antenna_delay_tx(const struct device *dev);

uint64_t dwt_rx_ts(const struct device *dev);

uint64_t dwt_system_ts(const struct device *dev);
uint32_t dwt_system_short_ts(const struct device *dev);

uint64_t dwt_ts_to_fs(uint64_t ts);
uint64_t dwt_fs_to_ts(uint64_t fs);

uint64_t dwt_short_ts_to_fs(uint32_t ts);
uint32_t dwt_fs_to_short_ts(uint64_t fs);

uint64_t dwt_calculate_actual_tx_ts(uint32_t planned_short_ts, uint16_t tx_antenna_delay);

void dwt_set_frame_filter(const struct device *dev, bool ff_enable, uint8_t ff_type);


int dwt_readcarrierintegrator(const struct device *dev);
float dwt_rx_clock_ratio_offset(const struct device *dev);

uint8_t *dwt_get_mac(const struct device *dev);

#endif /* ZEPHYR_INCLUDE_DRIVERS_IEEE802154_DW1000_H_ */
