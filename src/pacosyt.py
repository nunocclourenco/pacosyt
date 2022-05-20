#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import pickle
import wx
import wx.adv
import wx.py.shell

import windows as wd
from models import PassivesModel

''' VERSION INFO '''

TOOL_NAME = "PACOSYT"

TOOL_NAME_LONG = "PACOSYT: a Machine Learning based for PAssive COmponent SYthesis Tool"

TOOL_DESC = """PACOSYT is free and open source  advanced transformer and inductor 
modeling and synthesis tool using widely adopted ML libraries."""

TOOL_VER = '0.1'

TOOL_REPO = 'https://github.com/nunocclourenco/pacosyt'

TOOL_LIC = """MIT License
Copyright (c) 2022 Instituto de Telecomunicações & IMSE-CSIC
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT 
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR 
OTHER DEALINGS IN THE SOFTWARE."""

TOOL_ICONS = ('./img/logo_large.png',
              './img/logo.png',
              './img/institutions.jpg',
              './img/pacosyt.png'
              )

TOOL_SHELL_INTRO = """
This is the PACOSYT shell. Available commands are:

    'create(datapath, **options)' - TBD Fits a model using the data from the data file.
    'load(filepath)' - Loads the model from the file.
    'simulate(**geometry)' - predict S-Param and LQ for specified geometry.
    'optimize(freq, objectives, constraints)' - finds the best geometry
    'save(sri = None, lq = None)' - saves the required sri and/or LQ response obtained from the current simulation.
"""


class PacosytFrame(wx.Frame):

    def __init__(self, *args, **kwds):
        
        # begin wxGlade: MyFrame.__init__
        kwds["style"] = kwds.get("style", 0) | wx.RESIZE_BORDER | wx.CAPTION | wx.CLIP_CHILDREN | wx.CLOSE_BOX | wx.MAXIMIZE_BOX | wx.MINIMIZE_BOX | wx.SYSTEM_MENU
        wx.Frame.__init__(self, *args, **kwds)
        self.SetTitle(TOOL_NAME_LONG)
        _icon = wx.NullIcon
        _icon.CopyFromBitmap(wx.Bitmap(TOOL_ICONS[1], wx.BITMAP_TYPE_ANY))
        self.SetIcon(_icon)
        self.SetBackgroundColour(wx.Colour(255, 255, 255))

        # Menu Bar
        self.frame_menubar = wx.MenuBar()
        wxglade_tmp_menu = wx.Menu()
        item = wxglade_tmp_menu.Append(wx.ID_ANY, "Open Model", "")
        self.Bind(wx.EVT_MENU, self.on_open_file, item)
        item = wxglade_tmp_menu.Append(wx.ID_ANY, "Save SRI", "")
        self.Bind(wx.EVT_MENU, self.on_save_as_file, item)
        item = wxglade_tmp_menu.Append(wx.ID_ANY, "Exit", "")
        self.Bind(wx.EVT_MENU, self.on_exit, item)
        self.frame_menubar.Append(wxglade_tmp_menu, "File")
        wxglade_tmp_menu = wx.Menu()
        item = wxglade_tmp_menu.Append(wx.ID_ANY, "About", "")
        self.Bind(wx.EVT_MENU, self.on_about, item)
        self.frame_menubar.Append(wxglade_tmp_menu, "Help")
        self.SetMenuBar(self.frame_menubar)
        # Menu Bar end


        # Components
        '''the model, the workhorse of the app'''
        self.passives = PassivesModel()

        ''' GUI windows'''
        self.info = wx.adv.AboutDialogInfo()
        self.info.SetIcon(wx.Icon(TOOL_ICONS[0], wx.BITMAP_TYPE_PNG))
        self.info.SetName(TOOL_NAME_LONG)
        self.info.SetVersion(TOOL_VER)
        self.info.SetDescription(TOOL_DESC)
        self.info.SetCopyright('(C) 2022 Instituto de Telecomunicações & IMSE-CSIC')
        self.info.SetWebSite(TOOL_REPO)
        self.info.SetLicence(TOOL_DESC)
        self.info.AddDeveloper('N. Lourenço, F. Passos')
        self.info.AddDocWriter('F. Passos, E. Roca, R. Martins, R. Castro-López, N. Horta, F. V. Fernández')

        self.geometric_param_ctrl = wd.GeometricParametersCtrl(self,wx.ID_ANY)
        self.opt_ctrl = wd.OptimizationCtrl(self, wx.ID_ANY)
        self.sim_ctrl = wd.SimulationCtrl(self, wx.ID_ANY)
        
        self.button_optimize = wx.Button(self, wx.ID_ANY, "Optimize")
        self.button_simulate = wx.Button(self, wx.ID_ANY, "Simulate")
        self.button_optimize.SetDefault()


        self.shell = wx.py.shell.Shell(self, wx.ID_ANY, size=wx.Size(800, 150), introText = TOOL_SHELL_INTRO)
        self.shell.interp.locals["passives"] = self.passives
        self.shell.interp.locals["load"] = self.load
        self.shell.interp.locals["optimize"] = self.optimize
        self.shell.interp.locals["simulate"] = self.simulate
        self.shell.interp.locals["save"] = self.save
        
    
        image = wx.Bitmap(TOOL_ICONS[2], wx.BITMAP_TYPE_ANY).ConvertToImage()
        image = image.Scale(459, 106, wx.IMAGE_QUALITY_HIGH)
        footer_img = wx.StaticBitmap(self, wx.ID_ANY, wx.Bitmap(image), size=(459, 106))


        image = wx.Bitmap(TOOL_ICONS[3])
        image = image.ConvertToImage().Scale(180, 120, wx.IMAGE_QUALITY_HIGH)
        self.device_img= wx.StaticBitmap(self, wx.ID_ANY, wx.Bitmap(image), size=(180, 180))
        

        #layout
        sizer = wx.GridBagSizer(8, 8)
       
        sizer.Add(self.geometric_param_ctrl, (0, 0), (2,1), flag=wx.EXPAND|wx.ALIGN_TOP|wx.ALIGN_LEFT)
        sizer.Add(self.device_img, (0, 1), flag=wx.EXPAND|wx.ALIGN_CENTRE)
        
        button_sizer = wx.BoxSizer(wx.VERTICAL)
        button_sizer.Add(self.button_optimize,flag=wx.EXPAND)
        button_sizer.Add(self.button_simulate,flag=wx.EXPAND )
        sizer.Add(button_sizer, (1, 1), flag=wx.EXPAND)

        sizer.Add(self.opt_ctrl, (2, 0), (1,2), flag=wx.EXPAND|wx.ALIGN_TOP|wx.ALIGN_RIGHT)

        sizer.Add(self.sim_ctrl, (0, 2),  (3,1), flag=wx.EXPAND)
        
        shell_sizer = wx.BoxSizer(wx.VERTICAL)
        shell_sizer.Add(wx.StaticText(self, label= "Shell:"),flag=wx.EXPAND)
        shell_sizer.Add(wx.StaticLine(self), flag=wx.EXPAND)
        shell_sizer.Add(self.shell, flag=wx.EXPAND)
        sizer.Add(shell_sizer, (3, 0), (1,3), flag=wx.EXPAND)
      
        sizer.Add(footer_img, (4,0), (1,3), flag=wx.EXPAND|wx.ALIGN_CENTRE_VERTICAL|wx.ALIGN_RIGHT)
        
        margin_sizer = wd.BoxSizerWithMargins(margin=12, sizer=sizer)
        self.SetSizerAndFit(margin_sizer)


        self.Bind(wx.EVT_BUTTON, self.on_optimize, self.button_optimize)    
        self.Bind(wx.EVT_BUTTON, self.on_simulate, self.button_simulate)    
        

    #commands

    def load(self, filename):
        ''' Loads a model and updates GUI '''
        self.passives.load(filename)
        for panel in [self.geometric_param_ctrl,self.opt_ctrl, self.sim_ctrl]:
            panel.set_ranges(**self.passives.model['ranges'])
            panel.set_device(self.passives.model['device'])

        bitmap = wx.Bitmap(f"./img/{self.passives.model['device']}.png")
        image = bitmap.ConvertToImage().Scale(180, 120, wx.IMAGE_QUALITY_HIGH)
        self.device_img.SetBitmap(wx.Bitmap(image))

        
    def optimize(self, **kw):
        ''' optimize the for targetr specifications.
            optimize(self,freq, objct=["q"], cnstr=[["l",0.30, 5]])
          '''
        opt_geom = self.passives.optimize(**kw)
        self.geometric_param_ctrl.set_values(**opt_geom)
        self.simulate(kw['freq'], **opt_geom)
        return opt_geom


    def simulate(self, freq=None, **kw):
        ''' 
          '''
        resp, resp_meta = self.passives.simulate(**kw)
        self.sim_ctrl.set_values(resp, resp_meta, freq)
        return resp, resp_meta

    def save(self, sri_fname=None, lq_fname=None):
        self.passives.save(sri_fname, lq_fname)


    def on_optimize(self, event):

        l = ",".join([f"{k} = {v}" for k,v in self.opt_ctrl.get_values().items()])

        self.shell.run(f"opt_geom = optimize({l})")

    def on_simulate(self, event):
        l = ",".join([f"{k} = {v}" for k,v in self.geometric_param_ctrl.get_values().items()])
        self.shell.run(f"resp, resp_meta = simulate({l})")
        


    def on_open_file(self, event):  # wxGlade: MyFrame.<event_handler>
        filename = ""  # Use  filename as a flag
        dlg = wx.FileDialog(self, message="Choose a model")
        if dlg.ShowModal() == wx.ID_OK:
             # get the new filename from the dialog
            filename = dlg.GetPath()
        dlg.Destroy()  # best to do this sooner than later
 
        if filename:
            self.shell.run(f"load('{filename}')")


    def on_save_as_file(self, event):  # wxGlade: MyFrame.<event_handler>
        filename = ""  # Use  filename as a flag
        dlg = wx.FileDialog(self, message="Select file name", style= wx.FD_SAVE)
        if dlg.ShowModal() == wx.ID_OK:
             # get the new filename from the dialog
            filename = dlg.GetPath()
        dlg.Destroy()  # best to do this sooner than later
 
        if filename:
            self.shell.run(f"save('{filename}')")

    def on_exit(self, event):  # wxGlade: MyFrame.<event_handler>
        wx.Exit()
        event.Skip()

    def on_about(self, event):  # wxGlade: MyFrame.<event_handler>
        wx.adv.AboutBox(self.info,self)
        event.Skip()




class MyApp(wx.App):
    def OnInit(self):
        self.frame = PacosytFrame(None, wx.ID_ANY, TOOL_NAME)
        self.SetTopWindow(self.frame)
        self.frame.Show()
        return True


if __name__ == "__main__":
    app = MyApp(0)
    app.MainLoop()

    
