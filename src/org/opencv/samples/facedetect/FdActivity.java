package org.opencv.samples.facedetect;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.ListIterator;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.imgproc.Imgproc;

import android.app.Activity;
import android.content.Context;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.SubMenu;
import android.view.View;
import android.view.View.OnTouchListener;
import android.widget.Toast;
import android.view.WindowManager;

public class FdActivity extends Activity implements OnTouchListener, CvCameraViewListener2 {

	private static final String TAG = "OCVSample::Activity";
	private Mat mRgba;
	private Mat mGray;
	private Point resolutionPoint;
	private ScanTool mOpenCvCameraView;
	
	// 臉部辨識
	private File mCascadeFile;
	private CascadeClassifier mJavaDetector;
	private DetectionBasedTracker mNativeDetector;
	private float mRelativeFaceSizeMin = 0.09f;
	private float mRelativeFaceSizeMax = 0.9f;
	private int mAbsoluteFaceSizeMin = 0;
	private int mAbsoluteFaceSizeMax = 0;
	private MenuItem mItemFace;
	private boolean faceFUN = false;
	private boolean faceFUNtmp = false;
	
	// 輪廓辨識
	private MenuItem mItemFindContours;
	private final int HYSTERESIS_THRESHOLD1 = 128;
	private final int HYSTERESIS_THRESHOLD2 = 255;
	private boolean findContoursFUN = false;
	private boolean findContoursFUNtmp = false;
	
	// 解析度
	private boolean onCameraViewStarted = true;
	private List<android.hardware.Camera.Size> mResolutionList;
	private android.hardware.Camera.Size resolution = null;
	private SubMenu mResolutionMenu;
	private MenuItem[] mResolutionMenuItems;

    private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");

                    // Load native library after(!) OpenCV initialization
                    System.loadLibrary("detection_based_tracker");

                    try {
                        // load cascade file from application resources
                        InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface);
                        File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
                        mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
                        FileOutputStream os = new FileOutputStream(mCascadeFile);

                        byte[] buffer = new byte[4096];
                        int bytesRead;
                        while ((bytesRead = is.read(buffer)) != -1) {
                            os.write(buffer, 0, bytesRead);
                        }
                        is.close();
                        os.close();

                        mJavaDetector = new CascadeClassifier(mCascadeFile.getAbsolutePath());
                        /* 官方  Bug！要加入下列這一行，否則 JAVA 臉部辨識失效 */
                        mJavaDetector.load(mCascadeFile.getAbsolutePath());
                        if (mJavaDetector.empty()) {
                            Log.e(TAG, "Failed to load cascade classifier");
                            mJavaDetector = null;
                        } else
                            Log.i(TAG, "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());

                        mNativeDetector = new DetectionBasedTracker(mCascadeFile.getAbsolutePath(), 0);

                        cascadeDir.delete();

                    } catch (IOException e) {
                        e.printStackTrace();
                        Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
                    }

                    mOpenCvCameraView.enableView();
                } 
                break;
                default:
                {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.face_detect_surface_view);
        mOpenCvCameraView = (ScanTool) findViewById(R.id.fd_activity_surface_view);
        mOpenCvCameraView.setVisibility(ScanTool.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat();
        mGray = new Mat();
        
		// 螢幕解析度設定
		if(onCameraViewStarted == true){
    		onCameraViewStarted = false;
	        mResolutionList = mOpenCvCameraView.getResolutionList();
	        for(int i=0; i<mResolutionList.size(); i++){
	        	if(mResolutionList.get(i).width == 640){
	        		resolution = mResolutionList.get(i);
	        		mOpenCvCameraView.setResolution(resolution);
	        		resolution = mOpenCvCameraView.getResolution();
	        		String caption = Integer.valueOf(resolution.width).toString() + "x" + Integer.valueOf(resolution.height).toString();
	        		Toast.makeText(this, caption, Toast.LENGTH_SHORT).show();
	        	}
	        }
        }
    }

    public void onCameraViewStopped() {
        mRgba.release();
        mGray.release();
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {   	
		mRgba = inputFrame.rgba();
		mGray = inputFrame.gray();
		resolutionPoint = new Point(mRgba.width(), mRgba.height());
		
		// 輪廓辨識
		if(findContoursFUN){
			setFindContoursFUN(mGray, mRgba);
		}
				
		// 臉部辨識
		if(faceFUN && mJavaDetector != null){
			setFaceFUN(mGray, mRgba);
		}
						
        return mRgba;
    }
    
	// 臉部辨識
	private Mat setFaceFUN(Mat mMatOrg, Mat mRgba){
		Mat mTmp = new Mat();
		mMatOrg.copyTo(mTmp);
		
		if (mAbsoluteFaceSizeMin == 0) {
			int height = mRgba.rows();
			if (Math.round(height * mRelativeFaceSizeMin) > 0) {
				mAbsoluteFaceSizeMin = Math.round(height * mRelativeFaceSizeMin);
				mAbsoluteFaceSizeMax = Math.round(height * mRelativeFaceSizeMax);
				if(mAbsoluteFaceSizeMax > height){
					mAbsoluteFaceSizeMax = height;
				}
			}
			mNativeDetector.setMinFaceSize(mAbsoluteFaceSizeMin);
		}
		
		MatOfRect faces = new MatOfRect();
		
		mJavaDetector.detectMultiScale(mTmp, faces, 1.1, 6, 2, // TODO: objdetect.CV_HAAR_SCALE_IMAGE
                new Size(mAbsoluteFaceSizeMin, mAbsoluteFaceSizeMin), new Size(mAbsoluteFaceSizeMax, mAbsoluteFaceSizeMax));
		
        Rect[] facesArray = faces.toArray();
        for (int i = 0; i < facesArray.length; i++){
        	Imgproc.rectangle(mRgba, facesArray[i].tl(), facesArray[i].br(), new Scalar(255, 0, 255, 255), 3);
        }         

		return mRgba;
	}
	
	// 輪廓辨識
	private Mat setFindContoursFUN(Mat mMatOrg, Mat mRgba){
		Mat mTmp = new Mat();
		mMatOrg.copyTo(mTmp);

		// 二值化
		Imgproc.threshold(mTmp, mTmp,
				100, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C | Imgproc.THRESH_BINARY);
		
		// 影像金字塔(縮小)
		Imgproc.pyrDown(mTmp, mTmp,
				new Size(mRgba.cols()/2, mRgba.rows()/2));	
		
		// 蝕刻
		Imgproc.erode(mTmp, mTmp,
				Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(7, 7)));
		
		// 邊緣偵測
		Imgproc.Canny(mTmp, mTmp,
				HYSTERESIS_THRESHOLD1, HYSTERESIS_THRESHOLD2, 3, false);
		
		// 影像金字塔(放大)
		Imgproc.pyrUp(mTmp, mTmp,
				new Size(mTmp.cols()*2, mTmp.rows()*2));
		
		// 找影像輪廓
		ArrayList<MatOfPoint> contours = new ArrayList<MatOfPoint>();
		Imgproc.findContours(mTmp, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE, new Point(0, 0));

		if(contours.size() != 0 && contours.size() < 500) {
			// 劃出輪廓線
			Imgproc.drawContours(mRgba, contours, -1, new Scalar(255, 255, 0, 255), 1);

			//For each contour found
			MatOfPoint2f approxCurve = new MatOfPoint2f();
			for (int i=0; i<contours.size(); i++) {
				//Convert contours(i) from MatOfPoint to MatOfPoint2f
				MatOfPoint2f contour2f = new MatOfPoint2f( contours.get(i).toArray() );

				//Processing on mMOP2f1 which is in type MatOfPoint2f
				double approxDistance = Imgproc.arcLength(contour2f, true)*0.02;

				Imgproc.approxPolyDP(contour2f, approxCurve, approxDistance, true);

				//Convert back to MatOfPoint
				MatOfPoint points = new MatOfPoint( approxCurve.toArray() );

				// Get bounding rect of contour
				Rect rect = Imgproc.boundingRect(points);
				
				// draw enclosing rectangle (all same color, but you could use variable i to make them unique)
				Imgproc.rectangle(mRgba, new Point(rect.x,rect.y), 
						new Point(rect.x+rect.width,rect.y+rect.height), new Scalar(0, 255, 0, 255), 2);
				
//				// 輪廓取矩形可自動調整大小
//				RotatedRect rect2 = Imgproc.minAreaRect(contour2f);
//				Point[] vertices = new Point[4];
//				rect2.points(vertices);
//				for (int j = 0; j < 4; j++) {					
//					Core.line(mRgba, vertices[j], vertices[(j + 1) % 4], new Scalar(0, 255, 0, 255), 2);					
//				}
			
			}
			// 找影像輪廓數量顯示
			Imgproc.putText(mRgba, String.valueOf(contours.size()), new Point(10, resolutionPoint.y - 15), 3, 1, new Scalar(255, 0, 0, 255), 2);
 			
		} else {
			// 找影像輪廓數量顯示
			Imgproc.putText(mRgba, String.valueOf(0), new Point(10, resolutionPoint.y - 15), 3, 1, new Scalar(255, 0, 0, 255), 2);
		}
		return mRgba;
	}

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "called onCreateOptionsMenu");
        
		// 臉部
		mItemFace = menu.add("Face");
		
		// 輪廓
		mItemFindContours = menu.add("FindContours");
		
		// 螢幕解析度
        mResolutionMenu = menu.addSubMenu("Resolution");
        mResolutionList = mOpenCvCameraView.getResolutionList();
        mResolutionMenuItems = new MenuItem[mResolutionList.size()];
        ListIterator<android.hardware.Camera.Size> resolutionItr = mResolutionList.listIterator();
        int idx = 0;
        while(resolutionItr.hasNext()) {
            android.hardware.Camera.Size element = resolutionItr.next();
            mResolutionMenuItems[idx] = mResolutionMenu.add(2, idx, Menu.NONE,
                    Integer.valueOf(element.width).toString() + "x" + Integer.valueOf(element.height).toString());
            idx++;
        }
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);
        
		if (item.getGroupId() == 2){
			int id = item.getItemId();
            android.hardware.Camera.Size resolution = mResolutionList.get(id);
            mOpenCvCameraView.setResolution(resolution);
            resolution = mOpenCvCameraView.getResolution();
            String caption = Integer.valueOf(resolution.width).toString() + "x" + Integer.valueOf(resolution.height).toString();
            Toast.makeText(this, caption, Toast.LENGTH_SHORT).show();
		}
		
		// 臉部
		if (item == mItemFace) {
			faceFUN = true;
			if(faceFUN != faceFUNtmp) {
				faceFUNtmp = faceFUN;
				Toast.makeText(this, "Face: true", Toast.LENGTH_SHORT).show();
			} else {
				faceFUN = false;
				faceFUNtmp = faceFUN;
				Toast.makeText(this, "Face: false", Toast.LENGTH_SHORT).show();
			}
		}
		
		// 輪廓
		if (item == mItemFindContours) {
			findContoursFUN = true;
			if(findContoursFUN != findContoursFUNtmp) {
				findContoursFUNtmp = findContoursFUN;
				Toast.makeText(this, "FindContours: true", Toast.LENGTH_SHORT).show();
			} else {
				findContoursFUN = false;
				findContoursFUNtmp = findContoursFUN;
				Toast.makeText(this, "FindContours: false", Toast.LENGTH_SHORT).show();
			}
		}        
        return true;
    }

	@Override
	public boolean onTouch(View arg0, MotionEvent arg1) {
		// TODO Auto-generated method stub
		return false;
	}
}