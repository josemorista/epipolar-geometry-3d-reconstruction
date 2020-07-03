import { Mat } from 'opencv4nodejs';

export interface I2dPoint {
  x: number;
  y: number;
}

export interface IMatchPoint {
  p1: I2dPoint;
  p2: I2dPoint;
}

type cvMat = Mat;

type ndMat = Array<Array<number>>;