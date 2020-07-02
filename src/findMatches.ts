import * as cv from 'opencv4nodejs';

interface IMatchPoint {
  p1: { x: number, y: number; };
  p2: { x: number, y: number; };
}

const matchFeatures = (
  img1: cv.Mat,
  img2: cv.Mat,
  detector: cv.FeatureDetector,
  matchFunc: (descs1: cv.Mat, descs2: cv.Mat) => cv.DescriptorMatch[],
  bestN = 8
) => {
  // detect keypoints
  const keyPoints1 = detector.detect(img1);
  const keyPoints2 = detector.detect(img2);

  // compute feature descriptors
  const descriptors1 = detector.compute(img1, keyPoints1);
  const descriptors2 = detector.compute(img2, keyPoints2);

  // match the feature descriptors
  const matches = matchFunc(descriptors1, descriptors2);

  // only keep good matches
  const bestMatches = matches.sort(
    (match1, match2) => match1.distance - match2.distance
  ).slice(0, bestN);

  // create an array with img coordinates for the matches
  let matchedPoints: Array<IMatchPoint> = [];
  bestMatches.forEach(match => {
    matchedPoints.push({
      p1: { x: keyPoints1[match.queryIdx].pt.x, y: keyPoints1[match.queryIdx].pt.y },
      p2: { x: keyPoints2[match.trainIdx].pt.x, y: keyPoints2[match.trainIdx].pt.y }
    });
  });

  // draw matches to debug
  const matchedImg = cv.drawMatches(
    img1,
    img2,
    keyPoints1,
    keyPoints2,
    bestMatches
  );
  //cv.imshowWait('Matched img', matchedImg);

  // return matchedPoints for both images
  return matchedPoints;

};

export const findMatches = (img1Path: string, img2Path: string) => {

  console.log('reading imgs');
  const img1 = cv.imread(img1Path);
  const img2 = cv.imread(img2Path);


  console.log('processing sift...');
  const siftMatchedPoints = matchFeatures(
    img1,
    img2,
    new cv.SIFTDetector({ nFeatures: 2000 }),
    cv.matchFlannBased
  );

  console.log('processing ORB...');
  const orbMatchedPoints = matchFeatures(
    img1,
    img2,
    new cv.ORBDetector(),
    cv.matchBruteForceHamming
  );

  return { siftMatchedPoints, orbMatchedPoints };
};
