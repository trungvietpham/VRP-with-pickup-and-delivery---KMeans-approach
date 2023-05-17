@echo off
title Port data to 2evrp and run
FOR /L %%i IN (1, 1, 23) DO (
    :: Port data
    C:/Users/trung/miniconda3/envs/project2/python.exe "d:/TaiLieuHocTap/DANO/source code/port_data/gendata.py" 1 "D:\TaiLieuHocTap\DANO\benchmark_data\p"%%i
    :: Run 2evrp 
    @REM ping 192.0.2.2 -n 1 -w 10000 > nul
    C:/Users/trung/miniconda3/envs/project2/python.exe "d:/TaiLieuHocTap/DANO/source code/src/main.py" "D:\TaiLieuHocTap\DANO\benchmark_data\p"_all.res
    copy "D:\TaiLieuHocTap\DANO\source code\input\correlation.json" "D:\TaiLieuHocTap\DANO\MDVRP_MHA-master\GA\data_official\new_correlation.json" /Y
    :: Run GA for NDVRP
    C:/Users/trung/miniconda3/envs/project2/python.exe "d:/TaiLieuHocTap/DANO/MDVRP_MHA-master/GA/main.py" "D:\TaiLieuHocTap\DANO\benchmark_data\p"%%i "D:\TaiLieuHocTap\DANO\benchmark_data\p"_all.res 1
    echo %%i
    @REM pause
)
pause
:: C:/Users/trung/miniconda3/envs/project2/python.exe "port_data/gendata.py" 1 "D:\TaiLieuHocTap\Năm 4 - Kỳ 2\ĐATN\benchmark_data\p3"
