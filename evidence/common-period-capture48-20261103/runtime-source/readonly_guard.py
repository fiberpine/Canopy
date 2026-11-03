class ReadOnlySession:
    def __init__(self,session):self.session=session;self.cache=set();self.calls=0
    def run(self,fetches,feed_dict=None,**kwargs):
        def flatten(value):
            if isinstance(value,(tuple,list)):
                for item in value:
                    for child in flatten(item):yield child
            else:yield value
        fs=list(flatten(fetches))
        for f in fs:
            op=getattr(f,'op',f);name=op.name
            if name not in self.cache:
                stack=[op];seen=set()
                while stack:
                    x=stack.pop()
                    if x.name in seen:continue
                    seen.add(x.name)
                    if any(z in x.type for z in ['Assign','Apply','Scatter','SaveV2','RestoreV2','Enqueue','Dequeue']):raise RuntimeError('D0 mutating fetch rejected: '+x.name+' '+x.type)
                    stack.extend([t.op for t in x.inputs]);stack.extend(x.control_inputs)
                self.cache.add(name)
        self.calls+=1;return self.session.run(fetches,feed_dict=feed_dict,**kwargs)
