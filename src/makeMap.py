#! /usr/bin/env python
from laser_line_extraction.msg import LineSegment,LineSegmentList
import pandas as pd
import os
import rospy

def callback(data):
	global Map,isSaved
	lines = data.line_segments
	if not isSaved:
		for line in lines:
			ro = line.radius
			theta = line.angle
			print(ro,theta)
			Map = Map.append({'ro':ro,'theta':theta},ignore_index=True)
		output_dir = os.path.join(os.getcwd(),'map.csv')
		Map.to_csv(output_dir)
		isSaved = True
		print("Saved to {}".format(output_dir))
		 
if __name__ == '__main__':
	Map = pd.DataFrame(columns=['ro','theta'])
	isSaved=False
	rospy.init_node('makeMap',anonymous=True)
	rospy.Subscriber("line_segments",LineSegmentList, callback)
	while not isSaved:
		continue
		
