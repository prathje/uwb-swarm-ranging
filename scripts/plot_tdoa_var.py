import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from base import pair_index


RX_NOISE_STD =  1.0e-09

c_in_air = 299702547.236
RESP_DELAY_S = 0.001

NODE_DRIFT_STD = 0.0 #10.0/1000000.0

RESP_RATIO = 1.0


def calc(n=1):

   total = np.repeat(0.01, repeats=n)
   db = np.repeat(0.01, repeats=n)

   eps_ra = np.random.normal(scale=RX_NOISE_STD, size=n)
   eps_da = np.random.normal(scale=RX_NOISE_STD, size=n)
   eps_rb = np.random.normal(scale=RX_NOISE_STD, size=n)
   eps_db = np.random.normal(scale=RX_NOISE_STD, size=n)

   return ((total+eps_ra+eps_da) / (total+eps_rb+eps_db)) * (db+eps_db)



def tof_with_known_drift(n=1):

   tof = np.repeat(10.0/c_in_air, repeats=n)
   da = RESP_DELAY_S
   db = RESP_DELAY_S*RESP_RATIO
   ra = db+2*tof
   rb = da+2*tof

   drift_a = np.random.normal(loc=1.0, scale=NODE_DRIFT_STD, size=n)


   eps_ra = np.random.normal(scale=RX_NOISE_STD, size=n)
   eps_da = np.random.normal(scale=RX_NOISE_STD, size=n)
   eps_rb = np.random.normal(scale=RX_NOISE_STD, size=n)
   eps_db = np.random.normal(scale=RX_NOISE_STD, size=n)


   #return 0.5*drift_a*(ra+eps_ra) - 0.5*drift_a*(db+eps_db)
   return 0.5*drift_a*ra+0.5*drift_a*eps_ra - 0.5*drift_a*db-0.5*drift_a*eps_db
   #return 0.5*drift_a*eps_ra - 0.5*drift_a*eps_db


def tof_with_unknown_drift_alt(n=1):
   tof = np.repeat(10.0 / c_in_air, repeats=n)

   drift_a = np.random.normal(loc=1.0, scale=NODE_DRIFT_STD, size=n)
   drift_b = np.random.normal(loc=1.0, scale=NODE_DRIFT_STD, size=n)

   delay_a = RESP_DELAY_S
   delay_b = RESP_DELAY_S*RESP_RATIO

   poll_tx = np.repeat(0, repeats=n)
   poll_rx = poll_tx+np.random.normal(loc=tof, scale=RX_NOISE_STD, size=n)

   response_tx = poll_rx+np.repeat(delay_b, repeats=n)
   response_rx = response_tx + np.random.normal(loc=tof, scale=RX_NOISE_STD, size=n)

   final_tx = response_rx +np.repeat(delay_a, repeats=n)
   final_rx = final_tx + np.random.normal(loc=tof, scale=RX_NOISE_STD, size=n)


   drifted_noisy_ra = drift_a*(response_rx-poll_tx)
   drifted_noisy_da = drift_a*(final_tx-response_rx)
   drifted_noisy_rb = drift_b*(final_rx-response_tx)
   drifted_noisy_db = drift_b*(response_tx-poll_rx)

   return 0.5 *  (drifted_noisy_ra - (drifted_noisy_db) * ((drifted_noisy_ra+drifted_noisy_da) / (drifted_noisy_rb+drifted_noisy_db)))

def tdoa_with_unknown_drift_alt(n=1):
   tof = np.repeat(10.0 / c_in_air, repeats=n)

   # This TDoA Example is very simplistic but does not matter for the SD computation (it would matter however for positioning!)
   tof_al = np.repeat(5.0 / c_in_air, repeats=n)
   tof_bl = np.repeat(5.0 / c_in_air, repeats=n)

   drift_a = np.random.normal(loc=1.0, scale=NODE_DRIFT_STD, size=n)
   drift_b = np.random.normal(loc=1.0, scale=NODE_DRIFT_STD, size=n)
   drift_l = np.random.normal(loc=1.0, scale=NODE_DRIFT_STD, size=n)

   delay_a = RESP_DELAY_S
   delay_b = RESP_DELAY_S * RESP_RATIO

   poll_tx = np.repeat(0, repeats=n)
   poll_rx = poll_tx + np.random.normal(loc=tof, scale=RX_NOISE_STD, size=n)

   response_tx = poll_rx + np.repeat(delay_b, repeats=n)
   response_rx = response_tx + np.random.normal(loc=tof, scale=RX_NOISE_STD, size=n)

   final_tx = response_rx + np.repeat(delay_a, repeats=n)
   final_rx = final_tx + np.random.normal(loc=tof, scale=RX_NOISE_STD, size=n)


   passive_poll_rx =  poll_tx + np.random.normal(loc=tof_al, scale=RX_NOISE_STD, size=n)
   passive_response_rx = response_tx + np.random.normal(loc=tof_bl, scale=RX_NOISE_STD, size=n)
   passive_final_rx = final_tx + np.random.normal(loc=tof_al, scale=RX_NOISE_STD, size=n)


   drifted_noisy_ra = drift_a * (response_rx - poll_tx)
   drifted_noisy_da = drift_a * (final_tx - response_rx)
   drifted_noisy_rb = drift_b * (final_rx - response_tx)
   drifted_noisy_db = drift_b * (response_tx - poll_rx)

   drifted_noisy_ml = drift_l * (passive_response_rx - passive_poll_rx)
   drifted_noisy_ml2 = drift_l * (passive_final_rx - passive_response_rx)

   return 0.5 * ((drifted_noisy_ml+drifted_noisy_ml2)/(drifted_noisy_ra+drifted_noisy_da)) * drifted_noisy_ra + 0.5 * ((drifted_noisy_ml+drifted_noisy_ml2)/(drifted_noisy_ra+drifted_noisy_da)) * ((drifted_noisy_ra+drifted_noisy_da)/(drifted_noisy_rb+drifted_noisy_db)) * drifted_noisy_db - drifted_noisy_ml

def tdoa_with_unknown_drift_alt_using_cfo(n=1):
   tof = np.repeat(10.0 / c_in_air, repeats=n)

   # This TDoA Example is very simplistic but does not matter for the SD computation (it would matter however for positioning!)
   tof_al = np.repeat(5.0 / c_in_air, repeats=n)
   tof_bl = np.repeat(5.0 / c_in_air, repeats=n)

   drift_a = np.random.normal(loc=1.0, scale=NODE_DRIFT_STD, size=n)
   drift_b = np.random.normal(loc=1.0, scale=NODE_DRIFT_STD, size=n)
   drift_l = np.random.normal(loc=1.0, scale=NODE_DRIFT_STD, size=n)

   delay_a = RESP_DELAY_S
   delay_b = RESP_DELAY_S * RESP_RATIO

   poll_tx = np.repeat(0, repeats=n)
   poll_rx = poll_tx + np.random.normal(loc=tof, scale=RX_NOISE_STD, size=n)

   response_tx = poll_rx + np.repeat(delay_b, repeats=n)
   response_rx = response_tx + np.random.normal(loc=tof, scale=RX_NOISE_STD, size=n)

   final_tx = response_rx + np.repeat(delay_a, repeats=n)
   final_rx = final_tx + np.random.normal(loc=tof, scale=RX_NOISE_STD, size=n)

   passive_poll_rx = poll_tx + np.random.normal(loc=tof_al, scale=RX_NOISE_STD, size=n)
   passive_response_rx = response_tx + np.random.normal(loc=tof_bl, scale=RX_NOISE_STD, size=n)
   passive_final_rx = final_tx + np.random.normal(loc=tof_al, scale=RX_NOISE_STD, size=n)

   drifted_noisy_ra = drift_a * (response_rx - poll_tx)
   drifted_noisy_da = drift_a * (final_tx - response_rx)
   drifted_noisy_rb = drift_b * (final_rx - response_tx)
   drifted_noisy_db = drift_b * (response_tx - poll_rx)

   drifted_noisy_ml = drift_l * (passive_response_rx - passive_poll_rx)
   drifted_noisy_ml2 = drift_l * (passive_final_rx - passive_response_rx)

   return 0.5 * ((drifted_noisy_ml + drifted_noisy_ml2) / (
              drifted_noisy_ra + drifted_noisy_da)) * drifted_noisy_ra + 0.5 * (
                     (drifted_noisy_ml + drifted_noisy_ml2) / (drifted_noisy_ra + drifted_noisy_da)) * (drift_a/drift_b) * drifted_noisy_db - drifted_noisy_ml

   #return  0.5 * drift_a * ra + 0.5 * drift_a * eps_ra - 0.5 * drift_a * db * ((ra+eps_ra+da+eps_da) / (rb+eps_rb+db+eps_db)) - 0.5 * drift_a * eps_db * ((ra+eps_ra+da+eps_da) / (rb+eps_rb+db+eps_db))
   #return  0.5 * drift_a * eps_ra - 0.5 * drift_a * db * ((ra+eps_ra+da+eps_da) / (rb+eps_rb+db+eps_db)) - 0.5 * drift_a * eps_db * ((ra+eps_ra+da+eps_da) / (rb+eps_rb+db+eps_db))


NUM = 1000000

print(tof_with_known_drift(NUM).mean()*c_in_air)
print(tof_with_unknown_drift_alt(NUM).mean()*c_in_air)
print(tdoa_with_unknown_drift_alt(NUM).mean()*c_in_air)
#print(tdoa_with_unknown_drift_alt_using_cfo(NUM).mean()*c_in_air)

print(tof_with_known_drift(NUM).std()*c_in_air)
print(tof_with_unknown_drift_alt(NUM).std()*c_in_air)
print(tdoa_with_unknown_drift_alt(NUM).std()*c_in_air)
#print(tdoa_with_unknown_drift_alt_using_cfo(NUM).std()*c_in_air)