# Disabling swapping
is needed to save the life time of the drive.

* `sudo swapoff -a`
* `sudo rm -f /swapfile`
* `sudo systemctl mask swapfile.swap`
* `sudo systemctl disable mkswap.service`
* `echo -e "vm.swappiness = 0\nvm.overcommit_memory = 1\nvm.overcommit_ratio = 100\nvm.oom_kill_allocating_task = 1\nvm.panic_on_oom = 0\nvm.oom_dump_tasks = 1" | sudo tee -a /etc/sysctl.conf`
* `sudo sysctl -p`

With the last command you should get an output like:

```
vm.swappiness = 0
vm.overcommit_memory = 1
vm.overcommit_ratio = 100
vm.oom_kill_allocating_task = 1
vm.panic_on_oom = 0
vm.oom_dump_tasks = 1
```
