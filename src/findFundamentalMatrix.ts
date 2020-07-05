import path from 'path';
import { siftMatches } from './lib/findMatches';
import { Matrix, SVD } from 'ml-matrix';
import * as cv from 'opencv4nodejs';
import { eightPointAlgorithm } from './lib/eightPointAlgorithm';
import { ndMat, IMatchPoint, cvMat } from './@types';
import { matrix2ndMat } from './lib/utils';
import { ransac } from './lib/ransac';
import { getRandomInt } from './lib/utils';

const calculateRightEpipole = (F: ndMat) => {
  const Fmat = (new Matrix(F)).transpose();
  const svdF = new SVD(Fmat);
  return svdF.rightSingularVectors.getColumn(2);
};

const drawLines = (lines: Array<cv.Vec3>, img1: cvMat, img2: cvMat, pts1: Array<cv.Point2>, pts2: Array<cv.Point2>) => {
  const c = img1.cols;
  for (let i = 0; i < lines.length; i++) {
    const color = new cv.Vec3(getRandomInt(0, 255), getRandomInt(0, 255), getRandomInt(0, 255));
    let [x0, y0] = [0, -lines[i].z / lines[i].y].map(el => Math.round(el));
    let [x1, y1] = [c, -(lines[i].z + lines[i].x * c) / lines[i].y].map(el => Math.round(el));
    img1.drawLine(new cv.Point2(x0, y0), new cv.Point2(x1, y1), color, 1);
    img1.drawCircle(new cv.Point2(pts1[i].x, pts2[i].y), 5, color, -1);
    img2.drawCircle(new cv.Point2(pts2[i].x, pts2[i].y), 5, color, -1);
  }
};

const calculateSquareError = (sample: IMatchPoint, F: ndMat) => {
  const [x, y] = sample.p1, [xl, yl] = sample.p2;
  const xTmp = new Matrix([[x], [y], [1]]);
  let xlTmp = new Matrix([[xl, yl, 1]]);
  let M = xlTmp.mmul(new Matrix(F)).mmul(xTmp);
  let m = matrix2ndMat(M), sum = 0;
  for (let i = 0; i < m.length; i++) {
    for (let j = 0; j < m[i].length; j++) {
      sum += Math.pow(m[i][j], 2);
    }
  }
  return sum;
};

const img1 = cv.imread(path.resolve('.', 'datasets', 'temple', 'temple0001.png'));
const img2 = cv.imread(path.resolve('.', 'datasets', 'temple', 'temple0002.png'));

const matches = siftMatches(img1, img2, 1000);

let F = ransac(matches, 8, eightPointAlgorithm, calculateSquareError, 0.001, 0.95, 10);

const FcvMat = new cv.Mat(F, cv.CV_32F);
// const { F: FcvMat } = cv.findFundamentalMat(matches.map(el => new cv.Point2(el.p1.x, el.p1.y)), matches.map(el => new cv.Point2(el.p2.x, el.p2.y)), cv.FM_RANSAC);

const pts1 = matches.map(el => new cv.Point2(el.p1[0], el.p1[1]));
const pts2 = matches.map(el => new cv.Point2(el.p2[0], el.p2[1]));

const linesOnLeft = cv.computeCorrespondEpilines(pts2, 2, FcvMat);
const linesOnRight = cv.computeCorrespondEpilines(pts1, 1, FcvMat);

drawLines(linesOnLeft, img1, img2, pts1, pts2);
drawLines(linesOnRight, img2, img1, pts2, pts1);

cv.imwrite('result1.png', img1);
cv.imwrite('result2.png', img2);
