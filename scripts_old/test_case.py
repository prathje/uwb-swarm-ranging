import sys
import json

def convert_dist_to_ts(d):
    return (d / 299792458.0) / (15650.0*1e-15)

if __name__ == "__main__":

    p_a = 0.0
    p_b = 1.0
    p_l = 0.4

    DELAY_TS = 10
    DRIFT_A = 1.0002
    DRIFT_B = 1.0005
    DRIFT_L = 0.9991

    poll_tx_ts_a = 0
    poll_rx_ts_b = poll_tx_ts_a + convert_dist_to_ts(p_b-p_a)
    poll_rx_ts_l = poll_tx_ts_a + convert_dist_to_ts(p_l-p_a)

    response_tx_ts_b = poll_rx_ts_b + DELAY_TS
    response_rx_ts_a = response_tx_ts_b + convert_dist_to_ts(p_b-p_a)
    response_rx_ts_l = response_tx_ts_b + convert_dist_to_ts(p_b-p_l)

    final_tx_ts_a = response_rx_ts_a + DELAY_TS
    final_rx_ts_b = final_tx_ts_a + convert_dist_to_ts(p_b-p_a)
    final_rx_ts_l = final_tx_ts_a + convert_dist_to_ts(p_l-p_a)
    
    data_tx_ts_b = final_rx_ts_b + DELAY_TS
    
    
    def formatted_out(d, msg):
        sys.stdout.write(d + '\t' + json.dumps(msg) + '\n')

    formatted_out('A', {"type": "tx", "tx": {"addr": "A", "sn": 0, "ts": 0}, "rx": []})

    formatted_out('B', {"type": "tx", "tx": {"addr": "B", "sn": 0, "ts": 0}, "rx": []})

    # A polls B
    formatted_out('L', {"type": "rx", "clock_ratio_offset": 0, "tx": {"addr": "A", "sn": 1, "ts": poll_tx_ts_a*DRIFT_A}, "rx": []})

    # B responds
    formatted_out('L', {"type": "rx", "clock_ratio_offset": 0, "tx": {"addr": "B", "sn": 2, "ts": response_tx_ts_b*DRIFT_B}, "rx": [{"addr": "A", "sn": 1, "ts": poll_rx_ts_b*DRIFT_B}]})

    # A finalizes
    formatted_out('L', {"type": "rx", "clock_ratio_offset": 0, "tx": {"addr": "A", "sn": 3, "ts": final_tx_ts_a*DRIFT_A}, "rx": [{"addr": "B", "sn": 2, "ts": response_rx_ts_a*DRIFT_A}]})

    # B sends data
    formatted_out('L', {"type": "rx", "clock_ratio_offset": 0, "tx": {"addr": "B", "sn": 4, "ts": data_tx_ts_b*DRIFT_B}, "rx": [{"addr": "A", "sn": 3, "ts": final_rx_ts_b*DRIFT_B}, {"addr": "A", "sn": 1, "ts": poll_rx_ts_b*DRIFT_B}]})

    # L broadcasts its receptions
    formatted_out('L', {"type": "tx", "tx": {"addr": "L", "sn": 0, "ts": 100}, "rx": [{"addr": "A", "sn": 3, "ts": final_rx_ts_l*DRIFT_L}, {"addr": "B", "sn": 2, "ts": response_rx_ts_l*DRIFT_L}, {"addr": "A", "sn": 1, "ts": poll_rx_ts_l*DRIFT_L}]})