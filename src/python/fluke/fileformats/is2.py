#! /usr/bin/python
'''
Created on 20 Sep 2013

= Introduction =
================
Part of my-embedded-os is to generate tools for embedded development. One interesting example is the IS2 fileformat for monitoring PCB's temperature.

= Details =
===========
Two examples of what the code is capable of is in the samples folder.

Further Reading
http://www.irinfo.org/articles/article_4_2006_colbert.html

Vir beeldverwerking kyk na
http://matplotlib.org/users/image_tutorial.html
http://effbot.org/imagingbook/introduction.htm <---

Plotting:
http://stackoverflow.com/questions/12198264/image-and-mesh-in-same-plot-in-matlab
http://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html

@author: riaan
'''
#From Standard Python
import logging
import Image			#For Image Maniulation
import struct

#From myEmbeddedOSpythonLibs

class is2(object):
	'''
	classdocs
	'''
	
	def __init__(self, fileStream):
		self.fileStream = fileStream

		self.__vendor__ = None
		self.__modelNo__ = None
		self.__serialCameraNo__ = None
		self.__serialEngineNo__ = None
		self.__serialNo__ = None
		self.__FWver__ = None

		self.IR_data = None
		self.IR_raw = None
		self.emissivity = None
		self.transmission = None
		self.backgroundtemp = None
	
		self.Image1_data = None
		self.Image2_data = None

	def	calcTr(self,Tb, e = 0.9, t = 1.0, Te=20):
		"""
			Tr - body temperatures
			Tb - brillancy temperatures 
			e  - emissivity,
			t  - transmission, 
			Te - background (reflected) temperature
	
			Tb = e*t*Tr + (2-e-t)*Te
		"""
		if Tb is None:
			return None
		try:
			Tb_it = []
			if len(Tb) <= 1:
				Tb_it.append(Tb)
			else:
				Tb_it = Tb
		except:
			Tb_it = []
			Tb_it.append(Tb)
		Tr = []
		e = float(e)
		t = float(t)
		Te = float(Te)	
		for x in Tb_it:
			Tr.append((x - (2-e-t)*Te)/e*t)
		return Tr
	
	def	calcTb(self,Tr, e = 0.9, t = 1.0, Te=20):
		"""
			Tr - body temperatures
			Tb - brillancy temperatures 
			e  - emissivity,
			t  - transmission, 
			Te - background (reflected) temperature
	
			Tb = e*t*Tr + (2-e-t)*Te
		"""
		if Tr is None:
			return None
		try:
			Tr_it = []
			if len(Tr) <= 1:
				Tr_it.append(Tr)
			else:
				Tr_it = Tr
		except:
			Tr_it = []
			Tr_it.append(Tr)
		Tb = []
		e = float(e)
		t = float(t)
		Te = float(Te)
		for x in Tr_it:
			Tb.append(e*t*x + (2-e-t)*Te)
		return Tb
	 
	def doCalibration(self, measurement_unCal, e = 0.9, t = 1.0, Te=20):
					#Temperature_Tr, Measurement
	#	measurements = ((27.0, -37.75), 
	#				(36.6, 211.72), #TODO. Net seker maak dat agtergrond temp reg is
	#				(38.8, 250.75),
	#				(40.0, 268.5),
	#				(44.6, 320.75),
	#				(55.4, 418.0),
	#				(200.5, 4128.0),
	#				(1251.536388,  32000))  #Calculated, not calibrated
		measurements = ((0, -774.88), 
					(1024,  30142.8))  #Calculated, not calibrated
	
		Tb = []
		cal = []
		for mms in measurements:
			Tr = mms[0]
			Tb_res = self.calcTb(Tr,e,t,Te)[0]
			Tb.append(Tb_res)
			cal_res = mms[1]/Tb_res
			cal.append(mms[1]) 
			#print Tb_res, mms[1], cal_res
		#print Tb
		#print cal
		Tb_cal = np.interp(measurement_unCal, cal, Tb)
		return Tb_cal
	
	def typeConversion(self,value, fromType, toType):
		data = struct.pack(fromType,value)
		return struct.unpack(toType,data)[0]

	def readUnpack(self,l,fmt,infoToLog=None):
		if fmt == "S": #Assumes ASCII string
			res = self.fileStream.read(l)
			res.strip()
			logging.debug("S %s:", str(res))
		elif fmt == "Su": #ReadUnicodeString
			res = ""
			for i in range(0, int(l/2)):
				data = self.fileStream.read(2)
				res += chr(struct.unpack("H",data)[0])
			logging.debug("S %s:", str(res))
		elif fmt == "SH":
			res = self.strToHex(self.fileStream.read(l),"x")
			logging.debug("SH %s:", str(res))
		else:
			data = self.fileStream.read(l)
			res = struct.unpack(fmt,data)[0]
		if infoToLog is None:
			return res
		logging.info("%s %s",infoToLog,str(res))
		return res

	def read16bit_logOnly(self):
		data = []
		try:
			i = 0
			while(True):
				res = self.readUnpack(2,"<H")
				data.append(res)
				if (res < 128) and (res > 16):
					text = "\""+self.UINT16toStr(res)+"\""
				else:
					text = res
				a = i
				debugStr = "0x%04X"%a 
				debugStr += "(%03d)d:"%i 
				debugStr += " 0x%04X"%res 
				debugStr += " MSB:(%03d)d"%((res / 0x100) & 0xFF) 
				debugStr += " LSB:(%03d)d"%(res & 0xFF)
				debugStr += " >16b:(%05d)d"%res
				debugStr += " <16b:(%05d)d"%(((res & 0xFF)*0x100) | ((res / 0x100) & 0xFF))
				debugStr += " %03.3f"%self.UINT16toFloat(res)
				if (i > 1) and ((data[i-1] != 0) or (data[i] != 0)) and ((data[i-1] != 0xFF) and (data[i] != 0xFF)):
					debugStr += " %03.3f"%self.UINT32toFloat(data[i-1:i+1])
				debugStr += "\t"
				debugStr += str(text) 
				logging.debug(debugStr)
				i += 1
		except: 
			logging.debug("Error or EOF reached!")

	def read32bit_logOnly(self, single = True):
		#try:
		if (True):
			i = 0
			debugStr = ""
			while (single and (i == 0)):
				res = self.readUnpack(4,"<L")
				a = i
				if not (single):
					debugStr += "0x%04X"%a 
					debugStr += "(%03d)d:"%i 
				debugStr += " 0x%08X"%res 
				debugStr += " 32b:(%05d)d"%res
				debugStr += " 32f: %3.3f"%self.typeConversion(res, "<L", "f")
				debugStr += "\t"
				debugStr += str(res) 
				logging.debug(debugStr)
				i += 1
		#except: 
		#	logging.debug("Error or EOF reached!")

	def read(self):
		# Read Header
		self.fileStream.seek(0)
		
		for i in range(0, 0x38):
			self.read32bit_logOnly()

		#self.__FWver__ = self.UINT16ArraytoASCIIStr(data[64+o2:65+o2+1])

		self.__xIR_size__ = self.readUnpack(4,"<L","IR_xSize")
		self.__yIR_size__ = self.readUnpack(4,"<L","IR_ySize")
		self.__xImage_size__ = self.readUnpack(4,"<L","Image_xSize")
		self.__yImage_size__ = self.readUnpack(4,"<L","Image_ySize")
		
		for i in range(0, 9):
			self.read32bit_logOnly()
		#Vendor Data
		self.__vendor__ = self.readUnpack(64, "Su", "Vendor Name")
		self.__modelNo__ = self.readUnpack(64, "Su", "Model Number")
		self.__serialCameraNo__ = self.readUnpack(64, "Su", "Camera Serial Number")
		self.__serialEngineNo__ = self.readUnpack(64, "Su", "SW Engine Serial Number")
		self.__serialNo__ = self.__serialCameraNo__ + "," + self.__serialEngineNo__

		for i in range(0, 2):
			self.read32bit_logOnly()

		#Image Data:
		####################################
		self.__xImage1_size__ = self.readUnpack(4,"<L","Image1_xSize")
		self.__yImage1_size__ = self.readUnpack(4,"<L","Image1_ySize")

		for i in range(0, 21):
			self.read32bit_logOnly()
		self.__xImage1_size__ = self.readUnpack(4,"<L","Image1_xSize")
		self.__yImage1_size__ = self.readUnpack(4,"<L","Image1_ySize")

		self.Image1_data = []
		for i in range(self.__yImage1_size__):
			row = []
			rowText = "" 
			for j in range(0, self.__xImage1_size__):
				res = self.readUnpack(2,"H")
				row.insert(0, res)
				rowText += "%02X"%((res/0x100)&0xFF)
			logging.debug(rowText)
			self.Image1_data.append(row)

		for i in range(0, 3):
			self.read32bit_logOnly()

		#Thermal Data:
		####################################
		self.__xIR1_size__ = self.readUnpack(4,"<L","IR1_xSize")
		self.__yIR1_size__ = self.readUnpack(4,"<L","IR1_ySize")

		for i in range(0, 4):
			self.read32bit_logOnly()
	
		self.emissivity = self.readUnpack(4,"f","Emissivity")
		self.transmission = self.readUnpack(4,"f","Transmission")
		self.backgroundtemp = self.readUnpack(4,"f","Back Ground Temperature")

		for i in range(0, 3):
			self.read32bit_logOnly()

		if (True):
			self.IR_raw = []
			self.IR_data = []
			for i in range(0, self.__yIR1_size__):
				IR_data_row = [] 
				IR_raw_row = []
				IR_raw_row_text = ""
				for j in range(0, self.__xIR1_size__):
					res = self.readUnpack(2,"h")
					IR_raw_row.insert(0,res)
					tb_uncal = float((res))
					t = self.calcTr(self.doCalibration(tb_uncal), self.emissivity, self.transmission, self.backgroundtemp)[0]
					IR_data_row.insert(0,t)
					#IR_raw_row_text += "%03.3f "%t
					IR_raw_row_text += "%04X "%res
				logging.debug(IR_raw_row_text)
				self.IR_data.append(IR_data_row)
				self.IR_raw.append(IR_raw_row)
		#Calculate Average Temperature
		#rowCol = (5,6,7,8,9,10)
		rowCol = (7,8)
		cAverage = 0.0
		for i in rowCol:
			for j in rowCol:
				cAverage += float(self.IR_data[i][j])
		cAverage  = cAverage/(len(rowCol)*len(rowCol))
#		cAverage = float(self.IR_data[7][7]+self.IR_data[8][8]+self.IR_data[8][7]+self.IR_data[7][8])/4.0
		#Calcaulate Average blackbody radiation
		tbAverage = 0.0
		for i in rowCol:
			for j in rowCol:
				tbAverage += float(self.IR_raw[i][j])
		tbAverage = tbAverage/(len(rowCol)*len(rowCol))
		#tbAverage = float(self.IR_raw[7][7]+self.IR_raw[8][8]+self.IR_raw[8][7]+self.IR_raw[7][8])/4.0
		logging.debug("Center Average: %3.3f",cAverage)
		logging.debug("TB Average: %3.3f",tbAverage)
		self.cAverage = cAverage
		self.tbAverage = tbAverage

		for i in range(0, 36):
			self.read32bit_logOnly()

		#Merged Image Data:
		####################################
		self.__xImage2_size__ = self.readUnpack(4,"<L","Image1_xSize")
		self.__yImage2_size__ = self.readUnpack(4,"<L","Image1_ySize")

		self.Image2_data = []
		for i in range(self.__yImage2_size__):
			row = []
			rowText = "" 
			for j in range(0, self.__xImage2_size__):
				res = self.readUnpack(2,"H")
				row.insert(0,res)
				rowText += "%02X"%((res/0x100)&0xFF)
			logging.debug(rowText)
			self.Image2_data.append(row)

		for i in range(0, 6):
			self.read32bit_logOnly()
		self.cTemperature = float(self.readUnpack(4,"<L","Center Temperature"))*0.01
		
		#Correct for temperature around new center temperature
		d = 0.0
		self.cCorrectedAverage = self.cAverage
		if (True):
			d = self.cTemperature - self.cAverage
			temp = []
			for i in range(0, self.__yIR1_size__):
				row = [] 
				for j in range(0, self.__xIR1_size__):
					row.append(self.IR_data[i][j])
				temp.append(row)
			del self.IR_data
			self.IR_data = []
			for i in range(0, self.__yIR1_size__):
				IR_data_row = [] 
				for j in range(0, self.__xIR1_size__):
					t = temp[i][j] + d
					IR_data_row.append(t)
				self.IR_data.append(IR_data_row)
			self.cCorrectedAverage = 0.0
			for i in rowCol:
				for j in rowCol:
					self.cCorrectedAverage += float(self.IR_data[i][j])
			self.cCorrectedAverage  = self.cCorrectedAverage/(len(rowCol)*len(rowCol))

		tmax = self.IR_data[0][0]
		tmin = self.IR_data[0][0]
		for y in range(0, self.__yIR1_size__):
			for x in range(0,self.__yIR1_size__):
				if self.IR_data[y][x] > tmax:
					tmax = self.IR_data[y][x] 
				if self.IR_data[y][x] < tmin:
					tmin = self.IR_data[y][x]
		logging.debug("Min: %3.3f",tmin)
		logging.debug("Max: %3.3f",tmax)
		self.tMin = tmin
		self.tMax = tmax

		return


	def getCalibratedIRdata(self, emissivity = None, transmission = None, backgroundtemp = None, offset=None):
		if emissivity is None:
			emissivity = self.emissivity			
		if transmission is None:
			transmission = self.transmission
		if backgroundtemp is None:
			backgroundtemp = self.backgroundtemp
		if offset is None:
			offset = self.cTemperature - self.cAverage
		if self.IR_raw is None:
			return None
		try:
			self.IR_data = []
			for IR_raw_row in self.IR_raw:
				#print IR_raw_row
				IR_data_row = [] 
				for res in IR_raw_row:
					tb_uncal = float((res))
					t = self.calcTr(self.doCalibration(tb_uncal), emissivity, transmission, backgroundtemp)[0]
					#if t >= 200.5:
					#	print "200.5deg+", t, tb_uncal, res
					IR_data_row.append(t+offset)
				self.IR_data.append(IR_data_row)
		except:
			return None
		return self.IR_data

	def getIRImageSize(self):
		if self.IR_data is None:
			return None
		if len(self.IR_data) <= 0:
			return None
		if self.IR_data[0] is None:
			return None
		if len(self.IR_data[0]) <= 0:
			return None
		x = len(self.IR_data)
		y = len(self.IR_data[0])
		return (x,y)

	def getImageData(self, theImageData):
		"""
		Images are coded in 16 bit as follows:
			r:g:b = 5:6:5
		returns the data as a float from 1.0 to 0.0
		"""
		if theImageData is None:
			return None
		#print theImageData
		imageData = []
		for y in theImageData:
			imageDataRow = [] 
			for x in y:
				r = (x/0x800)&0x01F
				g = (x/0x20)&0x03F
				b = (x&0x01F)
				r_s = int(r*1.0/(0x1F)*0xFF)
				g_s = int(g*1.0/(0x3F)*0xFF)
				b_s = int(b*1.0/(0x1F)*0xFF)
				#print "X %04X: %02X-%02X-%02X"%(x,r,r_s,b),
				imageDataRow.append((r_s,g_s,b_s))
			#print
			imageData.append(imageDataRow)
		return imageData

	def getImage1Data(self):
		return self.getImageData(self.Image1_data)

	def getImage2Data(self):
		return self.getImageData(self.Image2_data)

	def DataToPIL(self,data,targetSize=None, interp=Image.ANTIALIAS):
		# http://effbot.org/imagingbook/image.htm

		if targetSize is None:
			xTargetSize = self.__xImage1_size__
			yTargetSize = self.__yImage1_size__
			targetSize = (xTargetSize,yTargetSize)
		else:
			(xTargetSize,yTargetSize) = targetSize

		#Generate Image
		ySize = len(data)
		xSize = len(data[0])
		try:
			l = len(data[0][0])
		except:
			l = 1
		if l == 1:
			im = Image.new("L", (xSize, ySize))
		else:
			im = Image.new("RGB", (xSize, ySize))
		for y in range(0, ySize):
			for x in range(0,xSize):
				im.putpixel((x,y), data[x][y])
		
		#Check if resizeing is required
		if xSize == xTargetSize:
			if ySize == yTargetSize:
				return im
		
		im = im.resize((xTargetSize,yTargetSize), interp).transpose(Image.ROTATE_270)
		return im
	
	def PILtoData(self,im):
		imageData = []
		imageDataRow = []
		imNew = im.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_90) 
		(xTargetSize, yTargetSize) = imNew.size
		try:
			#Test for RGB
			pixel = imNew.getpixel((0,0))
			if len(pixel) == 3:
				l = 3
			else:
				l = 1
		except:
			l = 1
		for x in range(0,xTargetSize):
			imageDataRow = []
			for y in range(0, yTargetSize):
				pixel = imNew.getpixel((x,y))
				if l == 3:
					r = float(pixel[0])/255.0
					g = float(pixel[1])/255.0
					b = float(pixel[2])/255.0
					imageDataRow.append((r,g,b))
				else:
					imageDataRow.append(float(pixel)/255.0)
			imageData.append(imageDataRow)
		return imageData

if __name__ == '__main__':
	#The main function is used mainly for testing
	#--------------------------------------------
	#From Matplotlib
	import matplotlib.pyplot as plt
	import matplotlib.image as mpimg
	import matplotlib.pylab as pylab
	import numpy as np
	import scipy.ndimage
	import sys

	FORMAT = '%(asctime)-15s %(funcName)-15s %(levelname)s %(message)s'
	logging.basicConfig(level=logging.DEBUG, format=FORMAT)
	
	is2filename = None
	for i in sys.argv:
		if ".is2" in i:
			if is2filename is None:
				is2filename = []
			is2filename.append(i)

	if is2filename is None:
		is2filename = ["samples/vt_00008.is2"]
		#exit()

	for is2fileNameSingle in is2filename:
	#if (True):
		f = open(is2fileNameSingle,"rb")
		data = is2(f)
		data.read()
		IRdata = data.getCalibratedIRdata()
		Image1Data = data.getImage1Data()		
		Image2Data = data.getImage2Data()		
		f.close()
		del f

		#logging.debug("ImageData: %s", str(imageData))
		xTargetSize = 105*4
		yTargetSize = 105*4
		IRim = data.DataToPIL(IRdata,(xTargetSize,yTargetSize))
		#IRim.show()
		V1im = data.DataToPIL(Image1Data,(xTargetSize,yTargetSize))
		#V1im.show()
		V2im = data.DataToPIL(Image2Data,(xTargetSize,yTargetSize))
		#V2im.show()
		#IRdata = data.PILtoData(IRim)
		IRdata = np.fliplr(scipy.ndimage.interpolation.zoom(np.array(IRdata),105*4./16, order=1, prefilter=False))

		#Image2Data = np.fliplr(scipy.ndimage.interpolation.zoom(np.array(Image2Data),105*4./105))

		Image1Data = data.PILtoData(V1im)
		Image2Data = data.PILtoData(V2im)
		#Image2Data = data.PILtoData(V1im.convert("P"))#,palette=Image.ADAPTIVE))

		print "Center Temp %3.3f\xF8C"%data.cTemperature
		print "Average Center Temp %3.3f\xF8C"%data.cAverage
		print "Corrected Average Center Temp %3.3f\xF8C"%data.cCorrectedAverage
		print "RAW Center Average %3.3f / 0x%04X"%(data.tbAverage, int(data.tbAverage))
		print "Max Temp: %3.3f"%data.tMax
		print "Min Temp: %3.3f"%data.tMin
			
		#2D plot Simple
		if (False):
			im = pylab.imshow(IRdata, interpolation='bicubic')
			cb = plt.colorbar(im)
			plt.setp(cb.ax.get_yticklabels(), visible=False)
			plt.show()

		#3D simple contour
		if (True):
			from mpl_toolkits.mplot3d import Axes3D
			from matplotlib import cm
			#From: 
			#   http://matplotlib.org/mpl_examples/pylab_examples/contour_demo.py
			# and 
			#   http://matplotlib.org/api/pyplot_api.html?highlight=imshow#matplotlib.pyplot.imshow
			
			if (xTargetSize is None) or (yTargetSize is None):
				(xImage, yImage) = data.getIRImageSize()
			else:
				xImage = xTargetSize
				yImage = yTargetSize
			x = np.arange(0, float(xImage), 1)
			y = np.arange(0, float(yImage), 1)
			X, Y = np.meshgrid(x, y)
			
			fig = plt.figure(1,figsize=(19,6.5),dpi=100)
			fig.suptitle("Center Temp %3.3f oC\n"%data.cTemperature+
						"Average Center Temp %3.3f oC\n"%data.cAverage+
						"Corrected Average Center Temp %3.3f oC\n"%data.cCorrectedAverage+
						"RAW Center Average %3.3f / 0x%04X\n"%(data.tbAverage, int(data.tbAverage))+
						"Max Temp: %3.3f oC\n"%data.tMax+
						"Min Temp: %3.3f oC\n"%data.tMin,
						fontsize=9)
			#fig = plt.figure(1,figsize=(6,2),dpi=100)
			ax = fig.add_subplot(131, projection='3d')
			ax.plot_surface(X, -Y,IRdata, cmap=cm.jet)
			#cset = ax.contourf(X, Y, Image2Data, zdir='z', offset=0, cmap=cm.Spectral) #See http://wiki.scipy.org/Cookbook/Matplotlib/Show_colormaps 
			#CS = plt.contour(X, Y, imageData)
			#plt.clabel(CS, inline=1, fontsize=10)
			#im = pylab.imshow(imageData, interpolation='bicubic')
			#cb = plt.colorbar(im)
			#plt.setp(cb.ax.get_yticklabels(), visible=False)

			#2D simple contour
			#From: 
			#   http://matplotlib.org/mpl_examples/pylab_examples/contour_demo.py
			#   http://matplotlib.org/api/pyplot_api.html?highlight=imshow#matplotlib.pyplot.imshow
			# and 
			#   http://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes.contour
			plt.subplot(1,3,2)
			CS = plt.contour(X, Y, IRdata, cmap=cm.jet)
			plt.clabel(CS, inline=1, fontsize=10)
			#im = pylab.imshow(Image1Data.convert("L"), interpolation='bicubic')
			im = pylab.imshow(Image1Data)#, interpolation='bicubic')
			#cb = plt.colorbar(im)
			#plt.setp(cb.ax.get_yticklabels(), visible=False)

			plt.subplot(1,3,3)
			im2 = pylab.imshow(Image2Data)# ,cmap=cm.Spectral)#, interpolation='bicubic')
		
			fig.savefig(is2fileNameSingle+'.png', transparent=True)
			#fig.savefig(is2fileNameSingle+'.pdf', transparent=True)
			ax.patch.set_facecolor(fig.get_facecolor())
			
			if (True):
				if len(is2filename) == 1:
					plt.show()
					pass
			if (False):
				fc = open("tempLog.csv","a")
				fc.write(is2fileNameSingle)
				fc.write(", %3.3f\xF8C"%data.cTemperature)
				fc.write(", %3.3f\xF8C"%data.cAverage)
				fc.write(", %3.3f\xF8C"%data.tbAverage)
				for i in (4,5,6,7,8,9,10,11):
					for j in (4,5,6,7,8,9,10,11):
						fc.write(", %d"%data.IR_raw[i][j])
				fc.write("\n")
				fc.close()

	pass