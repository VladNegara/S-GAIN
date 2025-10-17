import psutil
import diskinfo

if __name__ == '__main__':
    print(psutil.disk_partitions())
    print(diskinfo.DiskInfo())