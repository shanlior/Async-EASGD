local colors = require 'ansicolors'

function printServer(string)
  if (type(string) == 'string') then
    print(colors('%{red}' .. string .. '%{reset}'))
  else
    print(string)
  end
end

function printClient(node,string)
  if (type(string) == 'string') then
    print(colors('%{blue}' .. "Client #" .. node .. ' '.. string .. '%{reset}'))
  else
    print(string)
  end
end
