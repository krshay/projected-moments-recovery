function nu_write_mrc(path,volume,voxel_size)
%NU_WRITE_MRC Write a 3-D array as an MRC map (mode 2, float32).
%   NU_WRITE_MRC(PATH,VOLUME) writes VOLUME so it can be opened in ChimeraX or
%   any other map viewer, which is the honest way to compare a reconstruction
%   against published figures: MATLAB's ISOSURFACE on a small grid produces a
%   faceted surface at whatever contour level the caller picked, whereas a map
%   viewer interpolates and chooses its level from the density distribution.
%   Rendering differences of that kind are easily mistaken for differences in
%   the reconstruction itself.
%
%   NU_WRITE_MRC(PATH,VOLUME,VOXEL_SIZE) sets the cell dimensions from the
%   voxel size in angstroms (default 1, i.e. arbitrary units).

validateattributes(volume,{'numeric'},{'3d','real','nonempty'});
if nargin < 3 || isempty(voxel_size), voxel_size = 1; end
volume = single(volume);
[nx,ny,nz] = size(volume);

fid = fopen(path,'w','ieee-le');
assert(fid >= 0,'Cannot open %s for writing.',path);
cleanup = onCleanup(@() fclose(fid));

fwrite(fid,[nx ny nz],'int32');            % 1-3   dimensions
fwrite(fid,2,'int32');                     % 4     mode 2 = float32
fwrite(fid,[0 0 0],'int32');               % 5-7   start
fwrite(fid,[nx ny nz],'int32');            % 8-10  intervals
fwrite(fid,voxel_size*[nx ny nz],'float32');% 11-13 cell dimensions
fwrite(fid,[90 90 90],'float32');          % 14-16 cell angles
fwrite(fid,[1 2 3],'int32');               % 17-19 axis order
fwrite(fid,[min(volume(:)) max(volume(:)) mean(volume(:))],'float32'); % 20-22
fwrite(fid,[0 0],'int32');                 % 23-24 ispg, nsymbt
fwrite(fid,zeros(1,25),'int32');           % 25-49 extra
fwrite(fid,[0 0 0],'float32');             % 50-52 origin
fwrite(fid,['MAP ' 68 68 0 0],'uint8');    % 53-54 magic + machine stamp
fwrite(fid,std(double(volume(:))),'float32');  % 55 rms
fwrite(fid,0,'int32');                     % 56    label count
fwrite(fid,zeros(1,200),'int32');          % 57-256 labels
here = ftell(fid);
assert(here == 1024,'MRC header is %d bytes, expected 1024.',here);
fwrite(fid,volume,'float32');
clear cleanup;
end
