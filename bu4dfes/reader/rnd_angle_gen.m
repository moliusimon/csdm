% Generate gaussian angle distribution
rng shuffle
r1y = normrnd(-1,0.58,80000,1); 
r2y = normrnd(1,0.58,80000,1);

ry = [r1y; r2y];
ry = ry(randperm(length(ry)));

ry = ry(ry>-1 & ry<1);
mu_idx = length(ry)/2;
yaw = ry(mu_idx-37500:mu_idx+37500-1);

rng shuffle
r1p = normrnd(-1,0.58,80000,1); 
r2p = normrnd(1,0.58,80000,1);

rp = [r1p; r2p];
rp = rp(randperm(length(rp)));

rp = rp(rp>-1 & rp<1);
mu_idx = length(rp)/2;
pitch = rp(mu_idx-37500:mu_idx+37500-1);

save('pitch_distro.mat', 'pitch')
save('yaw_distro.mat', 'yaw')

%% Visualize
% Angle histograms normated
figure(1)
h_y = histogram(yaw,180); hold on;
h_p = histogram(pitch,90);

ax = gca;
h = int8(hist3([yaw pitch], [180 90]));
h = imresize(h, [1000, 500]);
filt = fspecial('average', [50 50]);
h = imfilter(h, filt);

% Angle histograms 
figure(2)
colormap(hot);
imagesc(h');
xlabel('Yaw({\circ})', 'FontSize', 18);
ylabel('Pitch({\circ})', 'FontSize', 18);
set(gca,'XTick',1:166:1000);
set(gca,'YTick',1:83:1000);
set(gca,'XTickLabel',-90:30:90, 'fontsize',14);
set(gca,'YTickLabel',-45:15:45, 'fontsize',14);

% Angle distribution 
figure(3)
plot(-90:88, h_y.Values(1:end-1), -45:43, h_p.Values(1:end-1),'LineWidth', 3);
xlim([-100,100]);
ylim([0,1200]);
legend('YAW', 'PITCH');
ylabel('#samples','FontSize', 18);
xlabel('angle[degrees]','FontSize', 18);
ax = gca;
set(ax,'XTick',-90:10:90);


