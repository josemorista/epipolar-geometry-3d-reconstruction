import * as cv from 'opencv4nodejs';
import path from 'path';
import Stereo from './lib/3dReconstruction/Stereo';

const img1cvMat = cv.imread(path.resolve('.', 'datasets', 'teddy', 'im2.ppm'));
const img2cvMat = cv.imread(path.resolve('.', 'datasets', 'teddy', 'im6.ppm'));

const disparityRGB = Stereo.findDisparityMap(img1cvMat, img2cvMat, 64, 2, 'SSD');

cv.imwrite('disparity.png', new cv.Mat(disparityRGB, cv.CV_8UC3));
