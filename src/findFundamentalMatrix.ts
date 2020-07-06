import path from 'path';
import { siftMatches } from './lib/findMatches';
import { Matrix, SVD } from 'ml-matrix';
import * as cv from 'opencv4nodejs';
import { eightPointAlgorithm } from './lib/eightPointAlgorithm';
import { ndMat, IMatchPoint, cvMat, IPoint } from './@types';
import { matrix2ndMat } from './lib/utils';
import { ransac } from './lib/ransac';
import { getRandomInt } from './lib/utils';

const computeEpilines = (pts: Array<IPoint>, F: ndMat, transpose: boolean = false) => {
  let Fmat = new Matrix(F);
  if (transpose) {
    Fmat = Fmat.transpose();
  }
  let lines: Array<IPoint> = [];
  pts.forEach(p => {
    lines.push(Fmat.mmul(new Matrix([[...p, 1]]).transpose()).getColumn(0));
  });
  return lines;
};

const calculateRightEpipole = (F: ndMat) => {
  const Fmat = (new Matrix(F)).transpose();
  const svdF = new SVD(Fmat);
  return svdF.rightSingularVectors.getColumn(svdF.rightSingularVectors.columns - 1);
};

const drawLines = (lines: Array<IPoint>, img: cvMat) => {
  const c = img1.cols;
  for (let i = 0; i < lines.length; i++) {
    const color = new cv.Vec3(getRandomInt(0, 255), getRandomInt(0, 255), getRandomInt(0, 255));
    let [x0, y0] = [0, -lines[i][2] / lines[i][1]].map(el => Math.round(el));
    let [x1, y1] = [c, -(lines[i][2] + lines[i][0] * c) / lines[i][1]].map(el => Math.round(el));
    img.drawLine(new cv.Point2(x0, y0), new cv.Point2(x1, y1), color, 1);
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

let F = ransac(matches, 8, eightPointAlgorithm, calculateSquareError, 0.5, 0.95, 10);

const linesOnLeft = computeEpilines(matches.map(el => el.p1), F, true);
const linesOnRight = computeEpilines(matches.map(el => el.p1), F);

drawLines(linesOnRight, img2);
drawLines(linesOnLeft, img1);

cv.imwrite('resultAux1.png', img1);
cv.imwrite('resultAux2.png', img2);

// const { F: FcvMat } = cv.findFundamentalMat(matches.map(el => new cv.Point2(el.p1.x, el.p1.y)), matches.map(el => new cv.Point2(el.p2.x, el.p2.y)), cv.FM_RANSAC);