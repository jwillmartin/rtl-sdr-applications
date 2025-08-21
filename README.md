# Overview
This repo contains various RTL-based applications. Each application is contained in the [src](/src/) directory.

# Requirements
This repo was tested using a VM and Ubuntu 22.04. The following are what worked for me.
* rtl-sdr
```
sudo apt-get install -y rtl-sdr
sudo usermod -aG plugdev "$USER"
echo -e "blacklist dvb_usb_rtl28xxu\nblacklist rtl2832\nblacklist rtl2830" | \
  sudo tee /etc/modprobe.d/blacklist-rtl.conf
sudo modprobe -r dvb_usb_rtl28xxu rtl2832 rtl2830 2>/dev/null || true
sudo udevadm control --reload-rules
sudo udevadm trigger
```
* pyrtlsdr
```
pip3 install pyrtlsdr
```

**Note:** Additional dependencies are contained in each application's folder. 
