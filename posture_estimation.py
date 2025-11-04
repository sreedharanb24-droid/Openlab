import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fname = 'synthetic_imu_posture_100Hz_5min.csv'; 
imu = pd.read_csv(fname)
t  = imu.t_s.values 
ax = imu.ax_g.values
ay = imu.ay_g.values 
az = imu.az_g.values  
gx = imu.gx_dps.values
gy = imu.gy_dps.values
gz = imu.gz_dps.values
fs  = int(np.round(1/np.mean(np.diff(t)))) 
N   = t.size
pitch_acc = np.degrees(np.arctan2(-ax, np.sqrt(ay**2 + az**2)))
pitch_gyro = np.zeros(N)
for i in range(1, N):
    dt = t[i] - t[i-1]
    pitch_gyro[i] = pitch_gyro[i-1] + gy[i]*dt  # integrate deg/s

alpha = 0.98
pitch_deg = np.zeros(N)
pitch_deg[0] = pitch_acc[0]
for i in range(1, N):
    dt = t[i] - t[i-1]
    pitch_pred = pitch_deg[i-1] + gy[i]*dt
    pitch_deg[i] = alpha*pitch_pred + (1-alpha)*pitch_acc[i]

win = int(np.round(0.5*fs))    
b = np.ones(win)/win
pitch_filt_full = np.convolve(pitch_deg, b, mode='full')
pitch_filt = pitch_filt_full[:N]
safe_limit = 15;  # acceptable posture band (deg)
hyst = 2;     # hysteresis width (deg)
upper = safe_limit + hyst; # enter "bad" if above this
lower = safe_limit - hyst; # exit "bad" when below this

min_bad_s  = 2.0;   # must be bad for >= 2 s to trigger
min_bad_n  = int(np.round(min_bad_s*fs))
min_cue_s  = 1.0;    # keep cue ON at least 1 s
min_cue_n  = int(np.round(min_cue_s*fs))
vibrate = np.zeros(N, dtype=bool)
bad_cnt = 0; cue_cnt = 0; in_cue = False

for i in range(N):
    a = np.abs(pitch_filt[i])
    if not in_cue:
        if a > upper:
            bad_cnt = bad_cnt + 1
        elif a < lower:
            bad_cnt = 0
        else:
            bad_cnt = max(bad_cnt - 1, 0) 
        if bad_cnt >= min_bad_n:
            in_cue = True
            cue_cnt = min_cue_n
            vibrate[i] = True
            bad_cnt = 0
    else:
        if cue_cnt > 0:
            cue_cnt = cue_cnt - 1
            vibrate[i] = True
        else:
            if a < lower:
                in_cue = False
                vibrate[i] = False
            else:
                vibrate[i] = True

pct_time_safe = 100*np.mean(np.abs(pitch_filt) <= safe_limit)
rms_deg= np.sqrt(np.mean(pitch_filt**2))
n_cues = np.sum(np.diff(np.concatenate([[0], vibrate.astype(int)])) == 1)
print('%% Time in safe posture: %.1f %%' % (pct_time_safe))
print('RMS posture angle: %.2f deg' % (rms_deg))
print('Vibrotactile cues: %d' % (n_cues))

plt.figure(num='Pitch & Vibrotactile Cue')
plt.plot(t, pitch_deg, label='Pitch (complementary)', color=(0.9, 0.6, 0))
plt.plot(t, pitch_filt, 'b', linewidth=1.2, label='Pitch (filtered)')
plt.axhline(upper, linestyle='--', color='r'); plt.text(t[0], upper, 'Upper', va='bottom')
plt.axhline(lower, linestyle='--', color='r'); plt.text(t[0], lower, 'Lower', va='top')
scale = max(upper, np.max(np.abs(pitch_filt)))*0.6
plt.step(t, vibrate.astype(float)*scale, where='post', linewidth=1.2, label='Vibrate (scaled)', color='k')
plt.xlabel('Time (s)')
plt.ylabel('Angle (deg)')
plt.title('Estimated Pitch Angle & Vibrotactile Cue')
plt.grid(True)
plt.legend(loc='best')

plt.figure(num='Summary Metrics')
plt.bar([0,1,2], [pct_time_safe, rms_deg, n_cues])
plt.gca().set_xticks([0,1,2])
plt.gca().set_xticklabels(['% Time Safe','RMS Angle (deg)','Cues'])
plt.ylabel('Value')
plt.title('Summary')
plt.grid(True)

plt.show()