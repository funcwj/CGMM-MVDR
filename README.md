## Notes

The python version of scripts now perfermance badly(may have some logical errors), please refer to `apply_cgmm_beamforming.m`, which works ok.

### Usage

copy `run_cgmm_beamforming.sh` and `apply_cgmm_beamforming.m` to `local/`, then run
```shell
local/run_cgmm_beamforming.sh --nj 15 $chime4_data/data/audio/16kHz/isolated_6ch_track/ $enhancement_data
```
instead of baseline `local/run_beamform_6ch_track.sh` commands.

### Results
* 6ch

| Methods | dev-simu | dev-real | eval-simu | eval-real | avg |
|  :---:  |  :---:   |   :---:  |   :---:   |   :---:   | :---: |
|Beamformit(SAT)| 14.36%  | 12.99%   | 21.24%    | 21.55%    |  13.68%  |
|  CGMM(SAT)    | 12.65%  | 11.39%   | 20.18%    | 19.72%    |  12.02%  |




### Reference
T. Higuchi, N. Ito, T. Yoshioka, and T. Nakatani, "Robust mvdr beamforming using time-frequency masks for online/offline asr in noise," in ICASSP, 2016.
