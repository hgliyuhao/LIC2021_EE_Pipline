import fairies as fa

def search_distance(pattern, sequence):

    res = []
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            res.append(i)
    return res

def get_distance(sentence,word_1,word_2):

    position_1 = search_distance(word_1,sentence)
    position_2 = search_distance(word_2,sentence)

    # TODO 优化算法
    distance = len(sentence)*10
    for i in position_1:
        for j in position_2:
            add_distance = 0
            for k in range(min(i,j),max(i,j)):
                if sentence[k] == '，':
                    add_distance += 50
                elif sentence[k] == '。':
                    add_distance += 100
                elif sentence[k] == '｜':
                    add_distance += 100  
                elif sentence[k] == ',':
                    add_distance += 50
                    
            if abs(i-j) + add_distance < distance:
                distance = abs(i-j) + add_distance

    return distance 

def find_relation_for_submit(text,s_entries, o_entries):
    
    res = {}

    res['text'] = text
    res['spo_list'] = []
    
    predicate_count = {}
    for s_ in s_entries:
        s_type = s_[1][:s_[1].find('s_')]
        o_type = s_type + 'o_'

        if s_type not in predicate_count:
            predicate_count[s_type] = 1
        else:
            predicate_count[s_type] += 1

    complex_relation = ['上映时间','饰演','获奖','配音','票房']

    for s_ in s_entries:
        
        s_type = s_[1][:s_[1].find('s_')]
        o_type = s_type + 'o_'
        
        if predicate_count[s_type] == 1:
            
             # 判断是否为复杂关系
            if s_type in complex_relation:
                new = {}
                new['predicate'] = s_type
                new['subject'] = s_[0]
                new['subject_type'] = s_[1].replace(s_type,'').replace('s_','')
                new['object'] = {}
                new['object_type'] = {}
                for o_ in o_entries:
                    
                    if o_type in o_[1]:

                        o_value = o_[1].replace(o_type,'')

                        if s_type == '上映时间':
                            if o_value == '地点':
                                new['object']['inArea'] = o_[0]
                                new['object_type']['inArea'] = o_[1].replace(o_type,'')
                            else:
                                new['object']['@value'] = o_[0]
                                new['object_type']['@value'] = o_[1].replace(o_type,'')    
                        elif s_type == '饰演': 
                            if o_value == '影视作品':
                                new['object']['inWork'] = o_[0]
                                new['object_type']['inWork'] = o_[1].replace(o_type,'')
                            else:
                                new['object']['@value'] = o_[0]
                                new['object_type']['@value'] = o_[1].replace(o_type,'')    
                        elif s_type == '获奖':
                            if o_value == '作品':
                                new['object']['inWork'] = o_[0]
                                new['object_type']['inWork'] = o_[1].replace(o_type,'')
                            elif o_value == 'Date':
                                new['object']['onDate'] = o_[0]
                                new['object_type']['onDate'] = o_[1].replace(o_type,'')
                            elif o_value == 'Number':
                                new['object']['period'] = o_[0]
                                new['object_type']['period'] = o_[1].replace(o_type,'')
                            else:
                                new['object']['@value'] = o_[0]
                                new['object_type']['@value'] = o_[1].replace(o_type,'')    
                        elif s_type == '配音': 
                            if o_value == '影视作品':
                                new['object']['inWork'] = o_[0]
                                new['object_type']['inWork'] = o_[1].replace(o_type,'')
                            else:
                                new['object']['@value'] = o_[0]
                                new['object_type']['@value'] = o_[1].replace(o_type,'')    
                        elif s_type == '票房':
                            if o_value == '地点':
                                new['object']['inArea'] = o_[0]
                                new['object_type']['inArea'] = o_[1].replace(o_type,'')
                            else:
                                new['object']['@value'] = o_[0]
                                new['object_type']['@value'] = o_[1].replace(o_type,'')
                        
                        if new['object'] == {}:
                            new['object']['@value'] = o_[0]
                            new['object_type']['@value'] = o_[1].replace(o_type,'')
                
                if '@value' in new['object']:
                    res['spo_list'].append(new)
            else:

                for o_ in o_entries:
                    if o_type in o_[1]:

                        # 有实体需要写入提交文件中
                        # 新建字典
                        new = {}
                        new['predicate'] = s_type
                        new['subject'] = s_[0]
                        new['subject_type'] = s_[1].replace(s_type,'').replace('s_','')
                        new['object'] = {}
                        new['object_type'] = {}

                        o_value = o_[1].replace(o_type,'')

                        if s_type == '上映时间':
                            if o_value == '地点':
                                new['object']['inArea'] = o_[0]
                                new['object_type']['inArea'] = o_[1].replace(o_type,'')
                        elif s_type == '饰演': 
                            if o_value == '影视作品':
                                new['object']['inWork'] = o_[0]
                                new['object_type']['inWork'] = o_[1].replace(o_type,'')
                        elif s_type == '获奖':
                            if o_value == '作品':
                                new['object']['inWork'] = o_[0]
                                new['object_type']['inWork'] = o_[1].replace(o_type,'')
                            elif o_value == 'Date':
                                new['object']['onDate'] = o_[0]
                                new['object_type']['onDate'] = o_[1].replace(o_type,'')
                            elif o_value == 'Number':
                                new['object']['period'] = o_[0]
                                new['object_type']['period'] = o_[1].replace(o_type,'')
                        elif s_type == '配音': 
                            if o_value == '影视作品':
                                new['object']['inWork'] = o_[0]
                                new['object_type']['inWork'] = o_[1].replace(o_type,'')
                        elif s_type == '票房':
                            if o_value == '地点':
                                new['object']['inArea'] = o_[0]
                                new['object_type']['inArea'] = o_[1].replace(o_type,'')
                    
                        if new['object'] == {}:
                            new['object']['@value'] = o_[0]
                            new['object_type']['@value'] = o_[1].replace(o_type,'')
                        
                        if '@value' in new['object']:
                            res['spo_list'].append(new)

                    # merge slot
        else:
            
            new = {}
            new['predicate'] = s_type
            new['subject'] = s_[0]
            new['subject_type'] = s_[1].replace(s_type,'').replace('s_','')
            new['object'] = {}
            new['object_type'] = {}

            # 只在同类别的情况下计算距离最小的
            object_dict = {}
            for o_ in o_entries:
                if o_type in o_[1]:
                    ob_type = o_[1].replace(o_type,'')
                    # print(ob_type)
                    if ob_type not in object_dict:
                        object_dict[ob_type] = []
                        object_dict[ob_type].append(o_)
                    else:
                        object_dict[ob_type].append(o_)
            
            for ob_type in object_dict:

                min_distance = 9999
                nearest_entry = ''
                for o_ in object_dict[ob_type]:
                    distance = get_distance(text,s_[0],o_[0])
                    # 处理自己就是自己的客体
                    if distance < min_distance :
                        min_distance = distance
                        nearest_entry = o_

                o_ = nearest_entry
                if o_ != '':
                    o_value = o_[1].replace(o_type,'')
                    if s_type == '上映时间':
                        if o_value == '地点':
                            new['object']['inArea'] = o_[0]
                            new['object_type']['inArea'] = o_[1].replace(o_type,'')
                        else:    
                            new['object']['@value'] = o_[0]
                            new['object_type']['@value'] = o_[1].replace(o_type,'')    
                    elif s_type == '饰演': 
                        if o_value == '影视作品':
                            new['object']['inWork'] = o_[0]
                            new['object_type']['inWork'] = o_[1].replace(o_type,'')
                        else:    
                            new['object']['@value'] = o_[0]
                            new['object_type']['@value'] = o_[1].replace(o_type,'')      
                    elif s_type == '获奖':
                        if o_value == '作品':
                            new['object']['inWork'] = o_[0]
                            new['object_type']['inWork'] = o_[1].replace(o_type,'')
                        elif o_value == 'Date':
                            new['object']['onDate'] = o_[0]
                            new['object_type']['onDate'] = o_[1].replace(o_type,'')
                        elif o_value == 'Number':
                            new['object']['period'] = o_[0]
                            new['object_type']['period'] = o_[1].replace(o_type,'')
                        else:    
                            new['object']['@value'] = o_[0]
                            new['object_type']['@value'] = o_[1].replace(o_type,'')      
                    elif s_type == '配音': 
                        if o_value == '影视作品':
                            new['object']['inWork'] = o_[0]
                            new['object_type']['inWork'] = o_[1].replace(o_type,'')
                        else:    
                            new['object']['@value'] = o_[0]
                            new['object_type']['@value'] = o_[1].replace(o_type,'')      
                    elif s_type == '票房':
                        if o_value == '地点':
                            new['object']['inArea'] = o_[0]
                            new['object_type']['inArea'] = o_[1].replace(o_type,'')
                        else:    
                            new['object']['@value'] = o_[0]
                            new['object_type']['@value'] = o_[1].replace(o_type,'')      
                    else:
                        new['object']['@value'] = o_[0]
                        new['object_type']['@value'] = o_[1].replace(o_type,'')

            if '@value' in new['object']:
                    res['spo_list'].append(new)            

    return res

res = fa.read_json('entry_pointer.json')

final_res = []

for i in res:

    text = i[0]
    s_entries = []
    o_entries = []

    for e in i[1]:
        if 's' in e[1]:
            s_entries.append(e)
        else:
            o_entries.append(e)


    r = find_relation_for_submit(text,s_entries,o_entries)
    final_res.append(r)

fa.write_json('final_res.json',final_res)