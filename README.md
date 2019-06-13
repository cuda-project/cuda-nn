# 建立软连
/usr/bin/ld: 找不到 -lcufft
/usr/bin/ld: 找不到 -lpcap

sudo ln -s /usr/local/cuda/lib64/libcufft.so  /usr/local/lib/libcufft.so  
sudo ln -s /usr/lib/x86_64-linux-gnu/libpcap.so.0.8  /usr/local/lib/libpcap.so


# 创建build目录，切换到build目录，编译，连接，执行
mkdir build
cd build
cmake ..
make
./pro