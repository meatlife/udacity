#!/usr/bin/env python
# -*- coding: utf-8 -*-

#NOTICE:FOR 64-bit PYTHON3!

import xml.etree.cElementTree as ET
import re
import codecs
import json
from langconv import Converter

#检查osm里的k值种类和个数
def count_keys(osm_file):
    keys = {'count':0}
    for _, ele in ET.iterparse(osm_file):
        if ele.tag == 'tag':
            if ele.attrib['k'] in keys:
                keys[ele.attrib['k']] += 1
            else:
                keys[ele.attrib['k']] = 1
            keys['count'] +=1
    return keys

#读取json文件，返回列表
def read_json(json_file):
    data = []
    with codecs.open(json_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

#把列表写入json文件
def write_to_json(data, output_file):
    with codecs.open(output_file, 'w') as f:
        for node in data:
            f.write(json.dumps(node) + '\n')

#审核邮编postcode，返回不符合定义的node的列表
def audit_postcode(data):
    check_list = []
    for node in data:
        if  'address' in node and 'postcode' in node['address']:
            postcode = node['address']['postcode']
            if not (len(postcode) == 6 and postcode.isdigit() and postcode.startswith('510')):
                check_list.append(node)
    return check_list

#审核电话号码phone，修改为正规化格式(11位手机号码，或带区号11位座机号码)，返回不符合规则的node的列表
def audit_phone(data):
    check_list = []
    for node in data:
        if 'phone' in node:
            #去除符号和+86
            s = re.sub(r'\+86+|\D', '', node['phone'])
            if len(s) == 11 and (s.startswith('1') or s.startswith('020')) and node['phone'] != s:
                print('id:\'{}\':edit phone from \'{}\' to \'{}\''.format(node['id'],node['phone'],s))
                node['phone'] = s
            #如果座机号码不带区号020，则加上
            elif len(s) == 8 and node['phone'] != s:
                print('id:\'{}\':edit phone from \'{}\' to \'020{}\''.format(node['id'],node['phone'],s))
                node['phone'] = '020' + s
            else:
                check_list.append(node)
    return check_list

#审核名字name，利用names交叉对照，初步修改为正规化格式，返回不符合修改规则的node的列表
def audit_name(data):
    check_list = []
    for node in data:
        #如果没有name但有names.zh，则使name = names.zh，如果都没有，则添加到审核列表
        if not 'name' in node:
            if 'names' in node:
                if 'zh' in node['names']:
                    print('id:\'{}\':edit name to \'{}\''.format(node['id'],node['names']['zh']))
                    node['name'] = node['names']['zh']
                else:
                    check_list.append(node)
        else:
            if 'names' in node:
                #如果name由一串字符和names.en组成，则删去names.en部分
                if 'en' in node['names'] and node['name'].endswith(node['names']['en'])\
                and len(node['names']['en']) < len(node['name']):
                    edit = re.sub(node['names']['en'], '', node['name']).strip(' -.')
                    print('id:\'{}\':edit name from \'{}\' to \'{}\''.format(node['id'],node['name'],edit))
                    node['name'] = edit
                #如果name由names.zh和一串字符组成，则保留names.zh部分，
                #注意，这种格式转换的name并不能保证都是是names.zh+' '+names.en组成的格式，因此需要详细审阅edit logs
                elif 'zh' in node['names'] and node['name'].startswith(node['names']['zh'])\
                and len(node['names']['zh']) < len(node['name']) and re.search(r'[a-zA-Z]',node['name']):
                    print('id:\'{}\':edit name from \'{}\' to \'{}\''.format(node['id'],node['name'],node['names']['zh']))
                    node['name'] = node['names']['zh']
                #如果name里没有中文，则添加到审核列表
                elif not re.search(r'[\u4e00-\u9fa5]',node['name']):
                    check_list.append(node)
            elif not re.search(r'[\u4e00-\u9fa5]',node['name']):
                check_list.append(node)
    return check_list

#审核地址，返回不规范的node的列表，以及不规范的子字段内容的字典
def audit_address(data):
    check_list = []
    # 子字典如'province': { node['id'] : node['address']['province'], ...}
    check_address = {
        'province': {},
        'city': {},
        'district': {},
        'street': {},
        'housenumber': {}
    }
    #广州各区及县级市
    gz_district = ['越秀区', '海珠区', '荔湾区', '天河区', '白云区', '黄埔区', '花都区', '番禺区', '萝岗区', '南沙区', '从化市', '增城市']
    #道路名的后缀字
    street_character = ['路', '道', '街']
    #房号的后缀字
    housenumber_character = ['号', '室', '栋', '房', '座', '层', '楼', '铺']

    for node in data:
        if 'address' in node:
            if 'province' in node['address']:
                #如果省份province不是广东省
                if not node['address']['province'] == '广东省':
                    check_address['province'][node['id']] = node['address']['province']
                    check_list.append(node)
            if 'city' in node['address']:
                #如果城市city不是广州市
                if not node['address']['city'] == '广州市':
                    check_address['city'][node['id']] = node['address']['city']
                    check_list.append(node)
            if 'district' in node['address']:
                #如果区district不是广州的区
                if not node['address']['district'] in gz_district:
                    check_address['district'][node['id']] = node['address']['district']
                    check_list.append(node)
            if 'street' in node['address']:
                #如果街道名最后一个字不是正规的后缀
                if not node['address']['street'][-1] in street_character:
                    check_address['street'][node['id']] = node['address']['street']
                    check_list.append(node)
            if 'housenumber' in node['address']:
                #如果房号名最后一个字不是正规的后缀，或者不是数字
                if not (node['address']['housenumber'][-1] in housenumber_character or re.search(r'[0-9]',node['address']['housenumber'][-1])):
                    check_address['housenumber'][node['id']] = node['address']['housenumber']
                    check_list.append(node)
    return (check_list, check_address)

#利用zh_wiki.py和langconv.py来对name进行繁转简
def convert_cht_name(data):
    for node in data:
        if 'name' in node:
            edit = Converter('zh-hans').convert(node['name'])
            if node['name'] != edit:
                print('id:\'{}\':convert name from \'{}\' to \'{}\''.format(node['id'],node['name'],edit))
                node['name'] = edit

'''
if __name__ == '__main__':
    json_file = 'guangzhou_china.osm.json'
    output_file = 'audited_' + json_file
    data = read_json(json_file)
    postcode_check_list = audit_postcode(data)
    phone_check_list = audit_phone(data)
    name_check_list = audit_name(data)
    address_check_list, check_address = audit_address(data)
    convert_cht_name(data)
    write_to_json(data, output_file)
    print('Simply audited the file. Please review all the check lists, more works to do!')
'''