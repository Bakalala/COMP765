# Teaching CartPole

The repository includes single and double cartpole environments, starter code for linear quadratic regulator and interface to control cartpole envs.

## Dynamics data

Dynamics data can be accessed [through this onedrive link](https://mcgill-my.sharepoint.com/:f:/g/personal/amin_soleimaniabyaneh_mail_mcgill_ca/Ev6radWjGyxMpuz75JzAF7MBTC-Dr1izOCt3o6fFI1DqWg?e=W2fzFu). Updated versions of the data will be available with prior notice.

Note that you can also collect your customized data with the instructions provided below.


## CLI and main entry

The main simulation file is [cartpole_sim.py](cartpole_sim.py). You can use python to run the file with the described CLI args.

```
python3 cartpole_sim.py --env CartPole
```

The data collection mode can be activated by the following command for a single cartpole env.

```
python3 cartpole_sim.py --env CartPole --data-collection --dataset-name cartpole-dynamics-data
```

More CLI options are available if needed.

```
  --env ENV             Switch between single and double cartpole envs: CartPole, DoubleCartPole.

  --refresh-rate REFRESH_RATE
                        GUI refresh rate.

  --control-rate CONTROL_RATE
                        Control input will be applied once per timestep by default!

  --data-collection     Log data in a CSV file, later to be used for learning dynamics and more! This will override the control inputs.

  --sampling-rate SAMPLING_RATE
                        Samples are taken at each step by default.

  --reset-rate RESET_RATE
                        Resets the environment if data collection mode is active.

  --dataset-size DATASET_SIZE
                        Size of the collected dataset.

  --control-min CONTROL_MIN
                        Minimum control input for sampling from a uniform distribution.

  --control-max CONTROL_MAX
                        Maximum control input for sampling from a uniform distribution.

  --dataset-name DATASET_NAME
                        Pick a name for the collected dataset.
```

