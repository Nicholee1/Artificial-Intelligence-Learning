1. 退出账户，关闭cursor
2. 网站上settings注销账户
3. 重置MAC id
`curl -fsSL https://aizaozao.com/accelerate.php/https://raw.githubusercontent.com/yuaotian/go-cursor-help/refs/heads/master/scripts/run/cursor_mac_id_modifier.sh -o ./cursor_mac_id_modifier.sh && sudo bash ./cursor_mac_id_modifier.sh && rm ./cursor_mac_id_modifier.sh`
4. 重新打开cursor，打不开可能需要重新赋予owner权限：
`sudo chown -R $USER ~/Library/Application\ Support/Cursor  
sudo chown -R $USER ~/.cursor`