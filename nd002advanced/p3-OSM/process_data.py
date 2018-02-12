#!/usr/bin/env python
# -*- coding: utf-8 -*-

#NOTICE:FOR 64-bit PYTHON3!

import xml.etree.cElementTree as ET
import re
import codecs
import json
import pymongo
import audit_data

def shape_element(element):
    problemchars = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')
    CREATED = ["version", "changeset", "timestamp", "user", "uid"]

    if element.tag == "node" or element.tag == "way":
        node = {'created':{}}
        #先写入attrib，分为3类，分别是created里的，地理位置和其他如'id','visible'等
        for attrib_key in element.attrib:
            #写入created
            if attrib_key in CREATED:
                node['created'][attrib_key] = element.attrib[attrib_key]
            #写入地理位置
            elif attrib_key == 'lat' or attrib_key == 'lon':
                node['pos'] = [float(element.attrib['lat']),float(element.attrib['lon'])]
            else:
                node[attrib_key] = element.attrib[attrib_key]
        #再写入iter里面的，tag有<tag>和<nd>两种
        address = {}
        node_refs = []
        names = {}
        for iter_ele in element.iter():
            #tag = <tag>且k值不是乱码
            if iter_ele.tag == 'tag' and not problemchars.search(iter_ele.attrib['k']):
                keyword = iter_ele.attrib['k']
                #tag的k值是'addr:'开头且':'唯一
                if keyword.startswith('addr:') and keyword.count(':') == 1:
                    address[keyword[5:]] = iter_ele.attrib['v']
                #不是'addr:'开头的
                if not keyword.startswith('addr:'):
                    #把'name:'开头拆分为2级字典names
                    if keyword.startswith('name:'):
                        names[keyword[5:]] = iter_ele.attrib['v']
                    #简单把其他key里的':'换成'_'
                    elif ':' in keyword:
                        node[keyword.replace(':','_')] = iter_ele.attrib['v']
                    #防止键名冲突，污染数据
                    elif keyword == 'created':
                         node['former_created'] = iter_ele.attrib['v']
                    elif keyword == 'type':
                        node['former_type'] = iter_ele.attrib['v']
                    elif keyword == 'pos':
                        node['former_pos'] = iter_ele.attrib['v']
                    else:
                        node[keyword] = iter_ele.attrib['v']
            #处理tag = <nd>
            if iter_ele.tag == 'nd' and not problemchars.search(iter_ele.attrib['ref']):
                node_refs.append(iter_ele.attrib['ref'])
        #address和node_refs有值则写入
        if address != {}:
            node['address'] = address
        if node_refs != []:
            node['node_refs'] = node_refs
        if names != {}:
            node['names'] = names
        node['type'] = element.tag
        return node
    else:
        return None

# osm提取列表
def read_osm(osm_file):
    data = []
    for _, element in ET.iterparse(osm_file):
        el = shape_element(element)
        if el:
            data.append(el)
    return data

# 把列表写入json文件
def write_to_json(data, output_file):
    with codecs.open(output_file, 'w') as f:
        for node in data:
            f.write(json.dumps(node) + '\n')

#从osm写入到json
def osm_to_json(osm_file):
    json_file = "{0}.json".format(osm_file)
    with codecs.open(json_file, "w") as fo:
        for _, element in ET.iterparse(osm_file):
            el = shape_element(element)
            if el:
                fo.write(json.dumps(el) + "\n")

#从json写入到mongodb
def json_to_mongodb(json_file):
    client = pymongo.MongoClient('mongodb://localhost:27017')
    db = client.osm
    s = input('import {} to mongoDB?[y/n]'.format(json_file))
    if s == 'y':
        count_before = db.gz.find().count()
        print('documents before: {}'.format(count_before))
        with codecs.open(json_file, 'r') as f:
            for line in f:
                db.gz.insert_one(json.loads(line))
        count_after = db.gz.find().count()
        print('documents after: {}, imported {} documents'.format(count_after, count_after - count_before))
    else:
        print('Bye!')


if __name__ == '__main__':
    osm_file = 'sample.osm'
    output_file = 'audited_' + osm_file + '.json'

    data = read_osm(osm_file)
    #简单批量清洗数据
    audit_data.audit_postcode(data)
    audit_data.audit_phone(data)
    audit_data.audit_name(data)
    audit_data.audit_address(data)
    audit_data.convert_cht_name(data)
    write_to_json(data, output_file)
    #写入数据到mongodb
    #json_to_mongodb(output_file)