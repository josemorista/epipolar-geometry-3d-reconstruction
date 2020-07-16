import { ndMat, cvMat } from "../../../@types";
import Matrix from "ml-matrix";
import { matrix2ndMat } from "../../utils";
import { compareWindowsBySSd, getPixelWindow, compareWindowsByCorrelation } from "../epilines";

export const searchMatchInStereoEpiline = (img: ndMat, w: ndMat, row: number, errorFunction: 'SSD' | 'CORRELATION' = 'SSD') => {
  const errorsFunctions = {
    'SSD': compareWindowsBySSd,
    'CORRELATION': compareWindowsByCorrelation
  };
  let bestResult = {
    position: [] as Array<number>,
    error: Number.MAX_SAFE_INTEGER
  };
  for (let j = 0; j < img[row].length; j++) {
    const w2 = getPixelWindow(img, [row, j], w.length);
    if (w2) {
      const localError = errorsFunctions[errorFunction](w, w2);
      if (localError < bestResult.error) {
        bestResult.error = localError;
        bestResult.position = [row, j];
      }
    }
  }
  return bestResult.position;
};

const findDisparityMap = (img1cvMat: cvMat, img2cvMat: cvMat, maxDisparity: number, windowSize: number = 3, errorFunction: 'SSD' | 'CORRELATION' = 'SSD') => {
  const img1 = img1cvMat.getDataAsArray();
  const img2 = img2cvMat.getDataAsArray();
  let disparity = matrix2ndMat(Matrix.zeros(img1.length, img1[0].length));
  let max = Number.MIN_SAFE_INTEGER, min = Number.MAX_SAFE_INTEGER;

  const linearTransform = (x: number, min: number, max: number, a: number, b: number) => {
    return (b - a) * ((x - min) / (max - min)) + a;
  };

  for (let i = 0; i < img1.length; i++) {
    for (let j = 0; j < img1[i].length; j++) {
      const w = getPixelWindow(img1, [i, j], windowSize);
      if (w) {
        const matchPosition = searchMatchInStereoEpiline(img2, w, i, errorFunction);
        let disp = Math.abs(j - matchPosition[1]);
        if (disp < maxDisparity) {
          if (disp < min) {
            min = disp;
          }
          if (disp > max) {
            max = disp;
          }
          disparity[i][j] = disp;
        }
      }
    }
    console.log(i);
  }


  let disparityRGB = [] as Array<Array<Array<number>>>;

  for (let i = 0; i < disparity.length; i++) {
    disparityRGB.push([]);
    for (let j = 0; j < disparity[i].length; j++) {
      const color = Math.round(linearTransform(disparity[i][j], min, max, 1, 255));
      disparityRGB[i][j] = [color, color, color];
    }
  }

  return disparityRGB;
};


export default {
  findDisparityMap
};