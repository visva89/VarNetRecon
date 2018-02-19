nb = 2;

d = load('current_filters.mat');
% d = load('flts.mat');
Nl = size(d.xs, 1);
Nb = size(d.xs, 2);
fw = size(d.filters, 2);
Nf = size(d.filters, 5);
imsz = [size(d.xs, 3), size(d.xs, 4)];
plot(d.fv_test)
%% recons
Nshow = 7;
for i = 1 : Nshow
imgs = squeeze(d.xs(:, i, :, :));
imgs = permute(imgs, [2,3,1]);
subplot(Nshow, 1, i);
imagesc(abs(reshape(imgs, [imsz(1), imsz(2) * Nl]))); colormap gray; colorbar;
xticks([]); yticks([]);
daspect([1,1,1]);
caxis([0, 0.5])
if i == 1
    title('Layer', 'interpreter', 'latex');
end
end
%% filters
img = squeeze(d.filters(:, :, :, 1, :));
img = cat(3, img, ones([Nl, fw, 1, Nf])*max(img(:)) );
for i = 1 : Nl
    subplot(Nl, 1, i);
    
    tmp = reshape(img(i, :,:, :), [fw, (fw+1)*Nf]);
    imagesc(tmp); 
    xticks([]); yticks([]);
    ylabel(['L',num2str(i)])
    if i == 1
        title('Filters', 'interpreter', 'latex');
    end
    daspect([1,1,1])
%     colorbar;
end
colormap gray;
% img = reshape(img, [Nl, fw, fw*Nf]);

%% pot curves
img = squeeze(d.filters);
w = ceil(sqrt(Nl));
for i = 1 : Nl
    subplot(w, w, i);
    xln = linspace(-4.5, 4.5, size(d.interp_knots,2));
%     plot(xln, squeeze(d.interp_knots(i, :, :)));
    plot(xln, cumsum(squeeze(d.interp_knots(i, :, :)), 1), '-');
    
%     plot(squeeze(d.interp_knots(i, 1,:, :))');
    title(sprintf('Layer %d', i), 'interpreter', 'latex');
    xticks([]); yticks([]);
end

%% hist of fresp
for i = 1 : Nl
    subplot(Nl, 1, i);
    hist(fl(d.before_interp(i, 2:end,:,:,:)), 300)
end
%%
nl = 8;

for i = 1 : Nf
    subplot(2, Nf, i);
    imagesc(squeeze(d.before_interp(nl, nb, :,:, i)));
    xticks([]); yticks([]);
    subplot(2, Nf, i + Nf);
    imagesc(squeeze(d.after_interp(nl, nb, :,:, i)));
    xticks([]); yticks([]);
end