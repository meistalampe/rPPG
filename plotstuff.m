% filename = 'E:\GitHub\CovPySourceFile\Results\Results_ThermalData_18_06_2020_13_19_36.h5';
filename = 'E:\GitHub\CovPySourceFile\Results\Results_ThermalData_18_06_2020_13_24_58.h5';
h5disp(filename)
% info = h5info(filename);
periorbital = h5read(filename, '/PERIORBITAL');
maxillary = h5read(filename, '/MAXILLARY');
nose = h5read(filename, '/NOSETIP');

veri = h5read(filename, '/LMVERIFICATION');
veri_len = h5readatt(filename, '/LMVERIFICATION', 'n');

figure;
for i = 1:veri_len
    img = veri(:, :, i);
    imshow(img)
    pause(0.5)
end

fs = 5;
% filtered, np.ones((N,)) / N, mode='valid'
lf_po = lowpass(periorbital, 1, fs);
lf_ma = lowpass(maxillary, 1, fs);
lf_no = lowpass(nose, 1, fs);
N = 150;
ma_po = conv(lf_po, ones(1,N)/N, 'valid');
ma_ma = conv(lf_ma, ones(1,N)/N, 'valid');
ma_no = conv(lf_no, ones(1,N)/N, 'valid');

figure;
subplot(3,1,1);
plot(lf_po)
title('Subplot 1: Periorbital')
ylabel('Temperature [C]')
ylim([30 40])

subplot(3,1,2); 
plot(lf_ma);
title('Subplot 2: Maxillary')
ylabel('Temperature [C]')
ylim([30 40])

subplot(3,1,3); 
plot(lf_no);
title('Subplot 3: Nose')
ylabel('Temperature [C]')
xlabel('Sample')
ylim([30 40])
% get this from ds attr later

[b, a] = butter(3, [0.1 1]/fs, 'bandpass');
filt_po = filtfilt(b, a, periorbital);
filt_ma = filtfilt(b, a, maxillary);
filt_no = filtfilt(b, a, nose);

figure;
subplot(3,1,1);
plot(filt_po)
title('Subplot 1: Periorbital')
ylabel('Temperature [C]')

subplot(3,1,2); 
plot(filt_ma);
title('Subplot 2: Maxillary')
ylabel('Temperature [C]')

subplot(3,1,3); 
plot(filt_no);
title('Subplot 3: Nose')
ylabel('Temperature [C]')
xlabel('Sample')