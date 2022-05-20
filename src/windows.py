
import wx

from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas

from models.passives import plot_resp

class StaticBoxSizerWithMargins(wx.StaticBoxSizer):
    def __init__(self, box, margin=10, sizer = None):

        super().__init__(box, wx.HORIZONTAL)
        vsizer = wx.BoxSizer(wx.VERTICAL)
        if sizer :
            self.inner = sizer
        else:    
            self.inner = wx.BoxSizer(wx.VERTICAL)

        vsizer.AddSpacer(margin)
        vsizer.Add(self.inner,0, wx.EXPAND, 0)
        vsizer.AddSpacer(margin)

        super().AddSpacer(margin)
        super().Add(vsizer,0, wx.EXPAND, 0)
        super().AddSpacer(margin)
        

    def Add(self, *args, **kw):
        return self.inner.Add(*args, **kw)

    def AddSpacer(self, *args, **kw):
        return self.inner.AddSpacer(*args, **kw)

class BoxSizerWithMargins(wx.BoxSizer):
    def __init__(self, margin=10, sizer = None):

        super().__init__()
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        vsizer = wx.BoxSizer(wx.VERTICAL)
        if sizer :
            self.inner = sizer
        else:    
            self.inner = wx.BoxSizer(wx.VERTICAL)

        vsizer.AddSpacer(margin)
        vsizer.Add(self.inner,0, wx.EXPAND, 0)
        vsizer.AddSpacer(margin)

        hsizer.AddSpacer(margin)
        hsizer.Add(vsizer,0, wx.EXPAND, 0)
        hsizer.AddSpacer(margin)
        
        super().Add(hsizer,0, wx.EXPAND)

    def Add(self, *args, **kw):
        raise  NotImplementedError()
        

    def AddSpacer(self, *args, **kw):
        raise  NotImplementedError()


class GeometricParametersCtrl(wx.Panel):
    '''
    Panel that holds design parameters.
    '''
    parameters = ['np', 'dinp', 'wp', 'ns', 'dins', 'ws', 'nt', 'din', 'w']

    def __init__(self, parent, *args, **kw):
        super().__init__(parent, *args, **kw)
    
        #create all labels
        self.labels = {p:wx.StaticText(self, wx.ID_ANY, f"{p.capitalize()}: ", style=wx.ALIGN_CENTER_HORIZONTAL) for p in self.parameters[0:6]} 
        self.spin_ctrl = {p:wx.SpinCtrl(self, wx.ID_ANY, min=0, max=200) for p in self.parameters[0:6]}

        grid_sizer = wx.GridBagSizer(4, 4)

        sizer = StaticBoxSizerWithMargins(wx.StaticBox(self, wx.ID_ANY, " Geometric Parameters:"), sizer= grid_sizer)

        for line, label, spin in zip(range(len(self.labels.values())), self.labels.values(), self.spin_ctrl.values()):
            grid_sizer.Add(label, (line, 0), flag=wx.EXPAND|wx.ALIGN_CENTRE_VERTICAL|wx.ALIGN_RIGHT)
            grid_sizer.Add(spin,  (line, 1), flag=wx.EXPAND|wx.ALIGN_CENTRE_VERTICAL|wx.ALIGN_RIGHT)

        self.SetSizer(sizer)
        
        # register primary controler to be used for inductor
        for pt, pi in zip(self.parameters[0:3], self.parameters[6:9]):
            self.spin_ctrl[pi] = self.spin_ctrl[pt]
            self.labels[pi] = self.labels[pt]
        
    def get_values(self):
        return {k:v.GetValue() for k,v in self.spin_ctrl.items()}

    def set_device(self, device):
        for param in ['ns', 'dins', 'ws']:
            self.labels[param].Enable( device!= 'ind')
            self.spin_ctrl[param].Enable(device!= 'ind')

        to_update_label = ['W', 'Nt', 'Din'] if device == 'ind' else ['Wp', 'Np', 'Dinp']
        
        for param in to_update_label:
            self.labels[param.lower()].SetLabel(param + ": ")
            
    def set_values(self, **values):
        for k,v in values.items():
            if spin := self.spin_ctrl.get(k):
                spin.SetValue(v)

    def set_ranges(self, **ranges):
        for k,v in ranges.items():
            if spin := self.spin_ctrl.get(k):
                spin.SetMin(v[0])
                spin.SetMax(v[1])


class OptimizationCtrl(wx.Panel):
    '''
    Panel that holds the performance targets intervals.
    '''
    def _create_spec_l_windows(self, l):
        label = wx.StaticText(self, wx.ID_ANY, f"{l.capitalize()} = ", style=wx.TE_RIGHT)
        value = wx.TextCtrl(self, wx.ID_ANY, '0.3', style = wx.TE_RIGHT)
        tol = wx.TextCtrl(self, wx.ID_ANY, '5', style = wx.TE_RIGHT)
        
        self.tol[l] = tol
        self.label[l] = label
        self.value[l] = value

        return (label, value,
            wx.StaticText(self, wx.ID_ANY, " +/- ", style=wx.ALIGN_CENTER),
            tol, wx.StaticText(self, wx.ID_ANY, " % [nH] ", style=wx.ALIGN_CENTER_HORIZONTAL))

    def _create_spec_q_windows(self, q):
        label = wx.StaticText(self, wx.ID_ANY, f"{q.capitalize()}  > ", style=wx.TE_RIGHT)
        value = wx.TextCtrl(self, wx.ID_ANY, '10', style = wx.TE_RIGHT)
        
        self.label[q] = label
        self.value[q] = value

        return (label, value)


    def __init__(self, parent, *args, **kw):
        super().__init__(parent, *args, **kw)
    
        #create all labels
        self.label = {} 
        self.value = {}
        self.tol = {}

        grid_sizer = wx.GridBagSizer( 4, 4)

        sizer = StaticBoxSizerWithMargins(wx.StaticBox(self, wx.ID_ANY, " Find Device: "), 10, sizer = grid_sizer)
        
        self.label['freq'] = wx.StaticText(self, wx.ID_ANY, "Freq = ", style=wx.TE_RIGHT)
        self.value['freq'] = wx.SpinCtrl(self, wx.ID_ANY, '28',  min=0, max=200, style = wx.TE_RIGHT) 

        grid_sizer.Add(self.label['freq'], (0,0), (1,1), flag=wx.EXPAND|wx.ALIGN_CENTER)
        grid_sizer.Add(self.value['freq'], (0,1), (1,2), flag=wx.EXPAND|wx.ALIGN_LEFT)
        grid_sizer.Add(wx.StaticText(self, wx.ID_ANY, "[GHz]", style=wx.TE_LEFT), (0,3), (1,2), flag=wx.EXPAND|wx.ALIGN_CENTRE_VERTICAL| wx.ALIGN_LEFT)

        for i, win in enumerate(self._create_spec_l_windows('lp')):
            grid_sizer.Add(win, (1,i), (1,1), flag=wx.EXPAND|wx.ALIGN_CENTRE_VERTICAL|wx.ALIGN_RIGHT)
        
        for i, win in enumerate(self._create_spec_l_windows('ls')):
            grid_sizer.Add(win, (2, i), (1,1), flag=wx.EXPAND|wx.ALIGN_CENTRE_VERTICAL|wx.ALIGN_RIGHT)
        
        for i, win in enumerate(self._create_spec_q_windows('qp')):
            grid_sizer.Add(win, (3, i), (1,1), flag=wx.EXPAND|wx.ALIGN_CENTRE_VERTICAL|wx.ALIGN_RIGHT)

        for i, win in enumerate(self._create_spec_q_windows('qs')):
            grid_sizer.Add(win, (4, i), (1,1), flag=wx.EXPAND|wx.ALIGN_CENTRE_VERTICAL|wx.ALIGN_RIGHT)

        self.choice_objective = wx.Choice(self, wx.ID_ANY, choices=["Min. Area", "Max. Q"])
        self.choice_objective.SetSelection(1)
        self.on_choice_objective(None)

        grid_sizer.Add(wx.StaticText(self, wx.ID_ANY, "Target: ", style=wx.TE_RIGHT), (6, 0), flag=wx.ALIGN_CENTRE_VERTICAL|wx.ALIGN_RIGHT)
        grid_sizer.Add(self.choice_objective, (6, 1), (1,4), flag=wx.EXPAND|wx.CENTER)
        
        parent.Bind(wx.EVT_CHOICE, self.on_choice_objective, self.choice_objective)

        self.SetSizer(sizer)
        
        # register primary controler to be used for inductor
        self.label['l'] = self.label['lp']
        self.value['l'] = self.value['lp']
        self.tol['l'] = self.tol['lp']

        self.label['q'] = self.label['qp']
        self.value['q'] = self.value['qp']

    def set_device(self, device):
        for param in ['ls', 'qs']:
            self.label[param].Enable(device != 'ind')
            self.value[param].Enable(device != 'ind')
            if tol := self.tol.get(param):
                tol.Enable(device != 'ind')
        if device == "ind":
            self.label['l'].SetLabel("L = ")
            self.label['q'].SetLabel("Q  > ")
        else:
            self.label['lp'].SetLabel("Lp = ")
            self.label['qp'].SetLabel("Qp  > ")

        self.on_choice_objective(None)
        
    def get_values(self):
        r_val = {'freq':self.value['freq'].GetValue()}
        if self.label['ls'].IsEnabled():
            r_val['constraints'] = [ [c, float(self.value[c].GetValue()), float(self.tol[c].GetValue())] for c in ['lp','ls']]    
            r_val['objectives'] = [ 'qs', 'qp']
        else:
            r_val['constraints'] = [ [c, float(self.value[c].GetValue()), float(self.tol[c].GetValue())] for c in ['l']]    
            r_val['objectives'] = [ 'q']

        return r_val
        
    def set_ranges(self, **ranges):
        for k,v in ranges.items():
            if spin := self.value.get(k):
                spin.SetMin(v[0])
                spin.SetMax(v[1])

    def on_choice_objective(self, event):
        if self.choice_objective.GetSelection():
            self.label['qp'].Enable(False)
            self.value['qp'].Enable(False)
            self.label['qs'].Enable(False)
            self.value['qs'].Enable(False)
        else:
            self.label['qp'].Enable(True)
            self.value['qp'].Enable(True)
            if self.label['ls'].IsEnabled():
                self.label['qs'].Enable(True)
                self.value['qs'].Enable(True)

class MatplotlibCanvas(FigureCanvas):
    def __init__(self, parent, id=wx.ID_ANY, x_label=None, y_label=None):
        # 1x1 grid, first subplot
        figure = self.figure = Figure()
        self.axes = figure.add_subplot(111)

        self.x_label, self.y_label = (x_label, y_label)
        
        if x_label:
            self.axes.set_xlabel(x_label)
        
        if y_label:
            self.axes.set_ylabel(y_label)
        
        FigureCanvas.__init__(self, parent, id, figure)

    def clear(self):
        self.axes.clear()
        if self.x_label:
            self.axes.set_xlabel(self.x_label)
        
        if self.y_label:
            self.axes.set_ylabel(self.y_label)
        self.draw()

class SimulationCtrl(wx.Panel):

    captions = {
        'freq': 'Working\nFrequency (GHz)',
        'lp': 'Inductance\nPrimary (nH)',
        'ls': 'Inductance\nSecondary (nH)',
        'qp': 'Quality Factor\nPrimary',
        'qs': 'Quality Factor\nSecondary',
        'k':  'coupling \nk factor',
        'l': 'Inductance\n (nH)',
        'q': 'Quality Factor\n',
        
    }

    def __init__(self, parent, *args, **kw):
        super().__init__(parent, *args, **kw)

        #  components
        self.canvas_sim_l = MatplotlibCanvas(self, y_label="L (nH)",x_label="Frequency (GHz)")
        self.canvas_sim_l.SetMinSize((600, 200))
        
        self.canvas_sim_q = MatplotlibCanvas(self, y_label="Q", x_label="Frequency (GHz)")
        self.canvas_sim_q.SetMinSize((600, 200))
        
        self.slider = wx.Slider(self,value= 28, minValue = 0, maxValue = 200)
        self.labels = {}
        self.values = {} 
        
        grid_sizer = wx.GridBagSizer( 2, 2)
        sizer = StaticBoxSizerWithMargins(wx.StaticBox(self, wx.ID_ANY, " Simulation: "), 5, sizer = grid_sizer)

        grid_sizer.Add(self.canvas_sim_l, (0,0), (1,9), flag=wx.EXPAND)
        grid_sizer.Add(self.canvas_sim_q, (1,0), (1,9), flag=wx.EXPAND)
        
        for i, pf in enumerate(["freq", "lp", "ls", "qp", "qs", "k"]):
            self.labels[pf]  = wx.StaticText(self, wx.ID_ANY, self.captions[pf], style=wx.TE_CENTER)
            self.values[pf] = wx.TextCtrl(self, wx.ID_ANY, 'value', style=wx.TE_CENTRE | wx.TE_READONLY)
            grid_sizer.Add(self.labels[pf], (3, i+1), flag=wx.EXPAND)
            grid_sizer.Add(self.values[pf], (4, i+1), flag=wx.EXPAND)


        grid_sizer.Add(self.slider, (5,0), (1,9), flag=wx.EXPAND)
        self.SetSizerAndFit(sizer)
        parent.Bind(wx.EVT_SCROLL, self.on_slider_change, self.slider)


   

    def set_values(self, resp, resp_meta, freq=None):
        self.resp = resp
        self.resp_meta = resp_meta

        self.canvas_sim_l.clear()
        self.canvas_sim_q.clear()  

        plot_resp(resp, (self.canvas_sim_l.axes, self.canvas_sim_q.axes))
                 
        self.canvas_sim_l.draw()
        self.canvas_sim_q.draw()
        
        if freq:
            self.slider.SetValue(freq)
        
        self.on_slider_change(None)

    def clear(self):
        self.resp = None
        self.canvas_sim_l.clear()
        self.canvas_sim_q.clear()


    def set_ranges(self, **ranges):
        if freq_range := ranges.get('freq'):
            self.slider.SetRange(*freq_range)

    def set_device(self, device) :
        self.clear()
        for param in ['ls', 'qs', 'k']:
            self.labels[param].Enable(device != 'ind')
            self.values[param].Enable(device != 'ind')
       
        if device == "ind":
            self.labels['lp'].SetLabel(self.captions['l'])
            self.labels['qp'].SetLabel(self.captions['q'])
        else:
            self.labels['lp'].SetLabel(self.captions['lp'])
            self.labels['qp'].SetLabel(self.captions['qp'])

    def on_slider_change(self, event):
        if self.resp:
            fi = self.slider.GetValue()
            self.values['freq'].SetValue(f'{self.resp[fi][0]:.4}')
            lq  =  self.resp[fi][2] 
            if(lq.shape[1] == 2):
                self.values['lp'].SetValue(f'{lq[0,0]:.4}')
                self.values['qp'].SetValue(f'{lq[0,1]:.4}')
                self.values['ls'].SetValue('')
                self.values['qs'].SetValue('')
                self.values['k'].SetValue('')         
            else:
                self.values['lp'].SetValue(f'{lq[0,0]:.4}')
                self.values['qp'].SetValue(f'{lq[0,1]:.4}')
                self.values['ls'].SetValue(f'{lq[0,2]:.4}')
                self.values['qs'].SetValue(f'{lq[0,3]:.4}')
                self.values['k'].SetValue(f'{lq[0,4]:.4}')    



''' TEST APP for the windows '''
class TestFrame(wx.Frame):
    def __init__(self, *args, **kwds):
        # begin wxGlade: MyFrame.__init__
        kwds["style"] = kwds.get("style", 0) | wx.RESIZE_BORDER |  wx.CAPTION | wx.CLIP_CHILDREN | wx.CLOSE_BOX | wx.MAXIMIZE_BOX | wx.MINIMIZE_BOX | wx.SYSTEM_MENU
        wx.Frame.__init__(self, *args, **kwds)
        self.SetTitle("TEST")
        self.SetBackgroundColour(wx.Colour(255, 255, 255))

        
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.AddSpacer(10)
        sizer_v = wx.BoxSizer(wx.VERTICAL)
        sizer_v.AddSpacer(20)
        self.uut = GeometricParametersCtrl(self)
        sizer_v.Add(self.uut)
        sizer_v.AddSpacer(20)


        self.button1 = wx.Button(self, wx.ID_ANY, "B1")
        self.button1.SetDefault()
        sizer_v.Add(self.button1, 0, 0, 0)
        self.Bind(wx.EVT_BUTTON, self.on_button1, self.button1)

        self.button2 = wx.Button(self, wx.ID_ANY, "B2")
        self.button2.SetDefault()
        sizer_v.Add(self.button2, 0, 0, 0)
        self.Bind(wx.EVT_BUTTON, self.on_button2, self.button2)


        sizer_v.AddSpacer(20)
        sizer.Add(sizer_v)
        sizer.AddSpacer(10)
        self.SetSizer(sizer)
        sizer.Fit(self)
        self.Layout()

    def on_button1(self,event):
        self.uut.set_values(nt=1, w=15, din=156)

    
    def on_button2(self,event):
        self.uut.set_ranges(nt=(1,5), w=(4,15), din=(30, 180))


class MyApp(wx.App):
    def OnInit(self):
        self.frame = TestFrame(None, wx.ID_ANY, "TEST")
        self.SetTopWindow(self.frame)
        self.frame.Show()
        return True


if __name__ == "__main__":
    app = MyApp(0)
    app.MainLoop()