function volume = nu_load_volume(volume_path,grid_size,crop)
%NU_LOAD_VOLUME Load, crop, mask and resize a volume, matching the repo's
%   preprocessing so shell expansions are comparable with existing work.
%
%   NU_LOAD_VOLUME(PATH,GRID_SIZE,CROP) additionally discards the outer margin
%   of the map before masking and resizing. CROP is either a fraction in (0,1]
%   of the linear extent, kept about the centre, or an explicit [LO HI] index
%   range. CROP = 1 or omitted leaves the volume untouched and reproduces the
%   original behaviour exactly.
%
%   WHY IT MATTERS. A deposited map is a box much larger than the molecule, and
%   masking alone does not change that: it zeroes the margin but keeps the
%   array size, so after downsampling the molecule still occupies the same
%   small fraction of the output grid. Cropping instead spends the available
%   voxels on the molecule rather than on solvent.
%
%   Measured for 80S (a 360^3 map) at GRID_SIZE = 31, taking occupancy as the
%   fraction of voxels above a given fraction of the map maximum:
%
%       threshold      crop = 1     crop = 0.697
%         0.02*max       16.0%          16.9%
%         0.05*max        9.3%          10.4%
%         0.10*max        3.1%           5.2%
%
%   So the crop helps, but modestly, and by how much depends entirely on where
%   the threshold is put -- it is a 1.7x gain on the densest voxels and almost
%   nothing on the bulk. Do not expect it alone to turn a blob into a molecule;
%   in the L = 5 to L = 13 comparison the angular band limit dominated it.
%
%   The default 0.697 is the fraction matching krshay/Non-Uniform-Second-Moment,
%   which crops indices 50:300 of a 360^3 map. Note the fraction form keeps that
%   WIDTH but centres it, giving 55:305; pass the explicit range to match the
%   reference exactly.
%
%   It is not free: the crop changes the coefficient array, hence the recovery
%   problem. Runs made with and without it are not comparable.

if nargin < 3 || isempty(crop)
    crop = 1;
end

[~,~,extension] = fileparts(volume_path);
switch lower(extension)
    case '.mat'
        contents = load(volume_path);
        names = fieldnames(contents);
        volume = contents.(names{1});
    otherwise
        volume = ReadMRC_local(volume_path);
end
volume = double(real(volume));
n = min(size(volume));
centre = floor((size(volume)-n)/2);
volume = volume(centre(1)+(1:n),centre(2)+(1:n),centre(3)+(1:n));

if isscalar(crop)
    validateattributes(crop,{'numeric'},{'scalar','real','positive','<=',1});
    if crop < 1
        kept = max(1,round(crop*n));
        first = floor((n-kept)/2)+1;
        volume = volume(first:first+kept-1,first:first+kept-1,first:first+kept-1);
    end
else
    validateattributes(crop,{'numeric'},{'vector','numel',2,'integer','positive'});
    assert(crop(2) <= n && crop(1) < crop(2), ...
        'crop range [%d %d] does not fit a %d^3 volume.',crop(1),crop(2),n);
    volume = volume(crop(1):crop(2),crop(1):crop(2),crop(1):crop(2));
end
n = size(volume,1);
radius = (n-1)/2;
axis_points = (0:n-1)-radius;
[x,y,z] = ndgrid(axis_points,axis_points,axis_points);
volume(sqrt(x.^2+y.^2+z.^2) > radius) = 0;
if n ~= grid_size
    source = linspace(1,n,grid_size);
    [qx,qy,qz] = ndgrid(source,source,source);
    volume = interpn(1:n,1:n,1:n,volume,qx,qy,qz,'linear',0);
end
volume = volume/max(norm(volume(:)),eps);
end

function volume = ReadMRC_local(path)
fid = fopen(path,'r','ieee-le');
header = fread(fid,256,'int32=>int32');
nx = double(header(1)); ny = double(header(2)); nz = double(header(3));
mode = double(header(4));
fseek(fid,1024+double(header(24)),'bof');
switch mode
    case 0, data = fread(fid,nx*ny*nz,'int8=>double');
    case 1, data = fread(fid,nx*ny*nz,'int16=>double');
    case 2, data = fread(fid,nx*ny*nz,'single=>double');
    case 6, data = fread(fid,nx*ny*nz,'uint16=>double');
    otherwise, fclose(fid); error('Unsupported MRC mode %d',mode);
end
fclose(fid);
volume = reshape(data,nx,ny,nz);
end
