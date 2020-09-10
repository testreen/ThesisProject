import xml.etree.ElementTree as ET
from os.path import basename
from PIL import Image
import json
import xml.dom.minidom as md
import argparse
import os

def create_labelimg(position_array, class_array, filename):
    image_file = filename

    if image_file == '' or image_file is None:
        print('Please provide image file path')
        exit(0)

    class_types = ['inflammatory', 'lymphocyte', 'fibroblast and endothelial', 'epithelial']
    image = Image.open(image_file)
    i_width, i_height = image.size
    i_depth = image.layers
    image_name = os.path.basename(image_file)
    output_xml_file = 'labelIMG/'+image_name.split('.')[0] + '.xml'
    print(output_xml_file)

    labelimg_xml = ET.Element('annotation')
    folder = ET.SubElement(labelimg_xml, 'folder')
    folder.text = basename(os.getcwd())
    file_name = ET.SubElement(labelimg_xml, 'filename')
    file_name.text = image_name
    path = ET.SubElement(labelimg_xml, 'path')
    # path.text = os.getcwd() + str('\\') + image_name
    path.text = image_name
    source = ET.SubElement(labelimg_xml, 'source')
    s_child = ET.SubElement(source, 'database')
    s_child.text = 'Unknown'
    size = ET.SubElement(labelimg_xml, 'size')
    width = ET.SubElement(size, 'width')
    width.text = str(i_width)
    height = ET.SubElement(size, 'height')
    height.text = str(i_height)
    depth = ET.SubElement(size, 'depth')
    depth.text = str(i_depth)
    segmented = ET.SubElement(labelimg_xml, 'segmented')
    segmented.text = str(0)

    for i in range(0, len(class_array) - 1):
        object = ET.SubElement(labelimg_xml, 'object')
        name = ET.SubElement(object, 'name')
        name.text = class_types[int(class_array[i])]
        pose = ET.SubElement(object, 'pose')
        pose.text = 'Unspecified'
        truncated = ET.SubElement(object, 'truncated')
        truncated.text = str(0)
        difficult = ET.SubElement(object, 'difficult')
        difficult.text = str(0)
        bndbox = ET.SubElement(object, 'bndbox')
        x_min = ET.SubElement(bndbox, 'xmin')
        x_min.text = str(int(position_array[i][0]))
        y_min = ET.SubElement(bndbox, 'ymin')
        y_min.text = str(int(position_array[i][1]))
        x_max = ET.SubElement(bndbox, 'xmax')
        x_max.text = str(int(position_array[i][2]))
        y_max = ET.SubElement(bndbox, 'ymax')
        y_max.text = str(int(position_array[i][3]))

    xmlstr = ET.tostring(labelimg_xml).decode()
    newxml = md.parseString(xmlstr)
    with open(output_xml_file, 'w') as outfile:
        outfile.write(newxml.toprettyxml(indent='\t', newl='\n'))

    print('xml file created successfully : ' + output_xml_file)
