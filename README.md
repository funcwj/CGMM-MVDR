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

| Methods | dev-simu | dev-real | eval-simu | eval-real |
|  :---:  |  :---:   |   :---:  |   :---:   |   :---:   |
|Beamformit(SAT)| 14.36%  | 12.99%   | 21.24%    | 21.55%    |
|  CGMM(SAT)    | 11.38%  | 11.30%   | 15.34%    | 17.27%    |
|Beamformit(DNN)| 10.29%  | 9.59%   | 15.79%    | 16.73%    |
| CGMM(DNN) | 7.69%  | 8.40%   | 10.82%    | 13.51%    |
| Beamformit(sMBR) | 9.11%  | 8.46%   | 14.54%    | 15.07%    |
|  CGMM(sMBR)    | 6.88%  | 7.58%   | 10.15%    | 12.12%    |

### Reference
T. Higuchi, N. Ito, T. Yoshioka, and T. Nakatani, "Robust mvdr beamforming using time-frequency masks for online/offline asr in noise," in ICASSP, 2016.
